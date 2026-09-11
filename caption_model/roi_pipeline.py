#!/usr/bin/env python3
"""
roi_pipeline.py ── ROI 处理流水线

包含 ROI 处理的 4 个核心模块 + 显式去重:
  - ROIAlignPooler:        从 patches 上做 ROI Align (双线性插值)
  - ROINameAttrInjector:   把 attr 特征注入到 ROI tokens 上
  - PerROIQFormer:         每个 ROI 独立过 Q-Former 压缩
  - ROITokenAggregator:    把多 ROI 的 query tokens 聚合成全局 32q
  - ExplicitROIDeduplicator: 基于 IoU 的显式 ROI 去重
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .constants import EVA01_DIM, QFORMER_HIDDEN


# ============================================================
# ROIAlignPooler
# ============================================================
class ROIAlignPooler(nn.Module):
    """从 patches 序列上做 ROI Align (S×S 双线性插值)"""

    def __init__(self, roi_size: int = 7, grid: int = 16):
        super().__init__()
        self.roi_size = roi_size
        self.grid = grid

    def forward(self, patches, boxes_batch):
        B, P, D = patches.shape
        H = W = self.grid
        S = self.roi_size
        device = patches.device
        if P == H * W + 1:
            patches = patches[:, 1:]
        elif P != H * W:
            patches = F.interpolate(
                patches.permute(0, 2, 1).unsqueeze(0).float(),
                size=H * W, mode='linear', align_corners=False,
            ).squeeze(0).permute(0, 2, 1)
        N = max((len(b) for b in boxes_batch), default=1)
        out = torch.zeros(B, N, S * S, D, device=device, dtype=patches.dtype)
        grid_feat = patches.reshape(B, H, W, D)
        for b in range(B):
            boxes = boxes_batch[b]
            for i, box in enumerate(boxes):
                if i >= N:
                    break
                x1, y1, x2, y2 = box['bbox']
                xs = torch.linspace(x1 * W, x2 * W, S, device=device)
                ys = torch.linspace(y1 * H, y2 * H, S, device=device)
                x0 = xs.floor().long().clamp(0, W - 1)
                x1c = xs.ceil().long().clamp(0, W - 1)
                y0 = ys.floor().long().clamp(0, H - 1)
                y1c = ys.ceil().long().clamp(0, H - 1)
                wx = (xs - xs.floor()).to(patches.dtype)
                wy = (ys - ys.floor()).to(patches.dtype)
                gf = grid_feat[b]
                f00 = gf[y0.unsqueeze(1), x0.unsqueeze(0)]
                f01 = gf[y0.unsqueeze(1), x1c.unsqueeze(0)]
                f10 = gf[y1c.unsqueeze(1), x0.unsqueeze(0)]
                f11 = gf[y1c.unsqueeze(1), x1c.unsqueeze(0)]
                wy_ = wy.view(S, 1, 1)
                wx_ = wx.view(1, S, 1)
                tok = ((1 - wy_) * (1 - wx_) * f00 +
                       (1 - wy_) *      wx_  * f01 +
                            wy_  * (1 - wx_) * f10 +
                            wy_  *      wx_  * f11)
                out[b, i] = tok.reshape(S * S, D)
        return out


# ============================================================
# ROINameAttrInjector
# ============================================================
class ROINameAttrInjector(nn.Module):
    """把 attr_feat 注入到 ROI 49-token 上 (gated residual)"""

    def __init__(self, embed_dim=EVA01_DIM):
        super().__init__()
        D = embed_dim
        self.attr_proj = nn.Sequential(
            nn.LayerNorm(D * 2),
            nn.Linear(D * 2, D), nn.GELU(),
            nn.Linear(D, D), nn.LayerNorm(D),
        )
        self.gate = nn.Sequential(
            nn.LayerNorm(D * 2),
            nn.Linear(D * 2, 1), nn.Sigmoid(),
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, roi_49, attr_feat):
        if attr_feat is None:
            return roi_49
        B, N, S, D = roi_49.shape
        attr_exp = attr_feat.unsqueeze(2).expand(-1, -1, S, -1).float()
        roi_f = roi_49.float()
        cat = torch.cat([roi_f, attr_exp], dim=-1)
        gate_val = self.gate(cat)
        proj_out = self.attr_proj(cat)
        return (roi_f + gate_val * proj_out).to(roi_49.dtype)


# ============================================================
# PerROIQFormer
# ============================================================
class PerROIQFormer(nn.Module):
    """每个 ROI 独立过 Q-Former, 把 49 个 tokens 压缩成 Q 个 query"""

    def __init__(self, qformer):
        super().__init__()
        self.qformer = qformer
        self.num_queries = qformer.num_queries

    def forward(self, roi_49_feats):
        B, N, S, D = roi_49_feats.shape
        flat = roi_49_feats.reshape(B * N, S, D)
        compressed = self.qformer(flat.float())
        Q = compressed.shape[1]
        return compressed.reshape(B, N, Q, -1)


# ============================================================
# ROITokenAggregator
# ============================================================
class ROITokenAggregator(nn.Module):
    """
    把 (B, N, Q_in, D) 的多 ROI query tokens 聚合成 (B, Q_out, D)

    使用:
      - 学习的 importance 加权
      - 外部 importance_scores (置信度) 加权
      - is_core 加成 (1.5x)
      - 可学习的 aggregator queries (32q) 跨 attention
    """

    def __init__(self, embed_dim=QFORMER_HIDDEN, num_heads=8, num_queries=32):
        super().__init__()
        D = embed_dim
        Dh = D // num_heads
        self.num_heads = num_heads
        self.num_queries = num_queries
        self._scale = Dh ** -0.5

        self.agg_queries = nn.Parameter(torch.zeros(1, num_queries, D))
        nn.init.normal_(self.agg_queries, std=0.02)
        self.q_proj = nn.Linear(D, D, bias=False)
        self.k_proj = nn.Linear(D, D, bias=False)
        self.v_proj = nn.Linear(D, D, bias=False)
        self.out_proj = nn.Linear(D, D, bias=False)
        self.norm = nn.LayerNorm(D)
        self.importance_proj = nn.Sequential(
            nn.Linear(D, D // 4), nn.GELU(),
            nn.Linear(D // 4, 1), nn.Sigmoid(),
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, roi_q_tokens, importance_scores=None, is_core=None):
        B, N, Q_in, D = roi_q_tokens.shape
        Q_out = self.num_queries
        H = self.num_heads
        Dh = D // H
        device = roi_q_tokens.device

        roi_mean = roi_q_tokens.float().mean(dim=2)
        learned_imp = self.importance_proj(roi_mean).squeeze(-1)
        score_w = (importance_scores.float().to(device)
                   if importance_scores is not None
                   else torch.ones(B, N, device=device))
        if is_core is not None:
            core_boost = torch.where(
                is_core.to(device),
                torch.full((B, N), 1.5, device=device),
                torch.ones(B, N, device=device))
            score_w = score_w * core_boost
        final_w = F.softmax(learned_imp + score_w, dim=1)
        w_exp = final_w.unsqueeze(-1).unsqueeze(-1)

        weighted_kv = (roi_q_tokens.float() * w_exp).reshape(B, N * Q_in, D)
        queries = self.agg_queries.expand(B, -1, -1).float()
        Qq = self.q_proj(queries).reshape(B, Q_out, H, Dh).permute(0, 2, 1, 3)
        Kk = self.k_proj(weighted_kv).reshape(B, N * Q_in, H, Dh).permute(0, 2, 1, 3)
        Vv = self.v_proj(weighted_kv).reshape(B, N * Q_in, H, Dh).permute(0, 2, 1, 3)
        attn = F.softmax(
            torch.matmul(Qq, Kk.transpose(-2, -1)) * self._scale, dim=-1)
        ctx = torch.matmul(attn, Vv).permute(0, 2, 1, 3).reshape(B, Q_out, D)
        ctx = self.out_proj(ctx)
        out = self.norm(queries + ctx)
        return out.to(roi_q_tokens.dtype)


# ============================================================
# ExplicitROIDeduplicator
# ============================================================
class ExplicitROIDeduplicator(nn.Module):
    """基于 IoU + 同类别约束的 NMS 去重"""

    def __init__(self, iou_threshold=0.7, max_rois=13):
        super().__init__()
        self.iou_threshold = iou_threshold
        self.max_rois = max_rois

    def _box_iou(self, box1, box2):
        ix1 = max(box1[0], box2[0])
        iy1 = max(box1[1], box2[1])
        ix2 = min(box1[2], box2[2])
        iy2 = min(box1[3], box2[3])
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        a1 = max(1e-8, (box1[2] - box1[0]) * (box1[3] - box1[1]))
        a2 = max(1e-8, (box2[2] - box2[0]) * (box2[3] - box2[1]))
        return inter / (a1 + a2 - inter + 1e-8)

    def deduplicate(self, roi_feats, boxes_batch, is_core, importance_scores):
        B, M, D = roi_feats.shape
        device = roi_feats.device
        K = self.max_rois
        out_feats = torch.zeros(B, K, D, device=device, dtype=roi_feats.dtype)
        out_core = torch.zeros(B, K, dtype=torch.bool, device=device)
        out_scores = torch.zeros(B, K, device=device)

        for b in range(B):
            boxes = boxes_batch[b] if b < len(boxes_batch) else []
            n = min(len(boxes), M)
            if n == 0:
                continue
            scores_b = [
                boxes[i].get('confidence', 0.5) *
                (1.5 if (i < is_core.shape[1] and is_core[b, i].item()) else 1.0)
                for i in range(n)
            ]
            order = sorted(range(n), key=lambda i: -scores_b[i])
            keep = []
            suppressed = set()
            for idx in order:
                if idx in suppressed:
                    continue
                keep.append(idx)
                box_i = boxes[idx]['bbox']
                cat_i = boxes[idx].get('category_id', -1)
                for jdx in order:
                    if jdx in suppressed or jdx == idx:
                        continue
                    box_j = boxes[jdx]['bbox']
                    cat_j = boxes[jdx].get('category_id', -1)
                    if cat_i == cat_j and cat_i > 0:
                        if self._box_iou(box_i, box_j) > self.iou_threshold:
                            suppressed.add(jdx)
                if len(keep) >= K:
                    break
            for slot, idx in enumerate(keep[:K]):
                out_feats[b, slot] = roi_feats[b, idx]
                out_core[b, slot] = is_core[b, idx] if idx < is_core.shape[1] else False
                out_scores[b, slot] = (importance_scores[b, idx]
                                       if idx < importance_scores.shape[1] else 0.5)
        return out_feats, out_core, out_scores
