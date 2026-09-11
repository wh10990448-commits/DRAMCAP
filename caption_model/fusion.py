#!/usr/bin/env python3
"""
fusion.py ── 各类 Fusion 模块

包括:
  - AttrFinalGating:                Attribute → Visual 的最终门控融合
  - ExplicitImplicitFusion:         显式/隐式双向 cross-attention 融合
  - TriStreamFusion:                三流融合 (global / vis / implicit-name)
  - GlobalROIBidirectionalFusion:   全局-ROI 双向融合 (含 core / non-core 分桶)
  - ROICrossAttnRefinement:         ROI ← patches 的 cross-attention 精炼
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .constants import EVA01_DIM, QFORMER_HIDDEN


# ============================================================
# AttrFinalGating
# ============================================================
class AttrFinalGating(nn.Module):
    """vis_tokens ← obj_feat 的 cross-attention + sigmoid 门控"""

    def __init__(self, embed_dim=QFORMER_HIDDEN, num_heads=8):
        super().__init__()
        D = embed_dim
        Dh = D // num_heads
        self.num_heads = num_heads
        self.scale = Dh ** -0.5
        self.q_proj = nn.Linear(D, D, bias=False)
        self.k_proj = nn.Linear(D, D, bias=False)
        self.v_proj = nn.Linear(D, D, bias=False)
        self.out = nn.Linear(D, D, bias=False)
        self.gate = nn.Sequential(
            nn.LayerNorm(D * 2), nn.Linear(D * 2, 1), nn.Sigmoid(),
        )
        self.norm = nn.LayerNorm(D)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, vis_tokens, obj_feat):
        if obj_feat is None:
            return vis_tokens
        B, Nv, D = vis_tokens.shape
        No = obj_feat.shape[1]
        H, Dh = self.num_heads, D // self.num_heads
        Q = self.q_proj(vis_tokens.float()).reshape(B, Nv, H, Dh).permute(0, 2, 1, 3)
        K = self.k_proj(obj_feat.float()).reshape(B, No, H, Dh).permute(0, 2, 1, 3)
        V = self.v_proj(obj_feat.float()).reshape(B, No, H, Dh).permute(0, 2, 1, 3)
        attn = F.softmax(torch.matmul(Q, K.transpose(-2, -1)) * self.scale, dim=-1)
        ctx = torch.matmul(attn, V).permute(0, 2, 1, 3).reshape(B, Nv, D)
        ctx = self.out(ctx)
        gate_val = self.gate(torch.cat([vis_tokens.float(), ctx], dim=-1))
        out = self.norm(vis_tokens.float() + gate_val * ctx)
        return out.to(vis_tokens.dtype)


# ============================================================
# ExplicitImplicitFusion
# ============================================================
class ExplicitImplicitFusion(nn.Module):
    """显式 32q ↔ 隐式 8q 的双向 cross-attention 融合"""

    def __init__(self, embed_dim=QFORMER_HIDDEN, num_heads=8):
        super().__init__()
        D = embed_dim
        Dh = D // num_heads
        self.num_heads = num_heads
        self.head_dim = Dh
        self._scale = Dh ** -0.5

        # explicit ← implicit
        self.q_e2i = nn.Linear(D, D, bias=False)
        self.k_e2i = nn.Linear(D, D, bias=False)
        self.v_e2i = nn.Linear(D, D, bias=False)
        self.out_e = nn.Linear(D, D, bias=False)
        self.gate_e = nn.Sequential(
            nn.LayerNorm(D * 2), nn.Linear(D * 2, 1), nn.Sigmoid(),
        )
        self.norm_e = nn.LayerNorm(D)

        # implicit ← explicit
        self.q_i2e = nn.Linear(D, D, bias=False)
        self.k_i2e = nn.Linear(D, D, bias=False)
        self.v_i2e = nn.Linear(D, D, bias=False)
        self.out_i = nn.Linear(D, D, bias=False)
        self.gate_i = nn.Sequential(
            nn.LayerNorm(D * 2), nn.Linear(D * 2, 1), nn.Sigmoid(),
        )
        self.norm_i = nn.LayerNorm(D)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _attn(self, q_proj, k_proj, v_proj, out_proj, q_src, kv_src):
        B, Nq, D = q_src.shape
        Nkv = kv_src.shape[1]
        H = self.num_heads
        Dh = self.head_dim
        Q = q_proj(q_src).reshape(B, Nq, H, Dh).permute(0, 2, 1, 3)
        K = k_proj(kv_src).reshape(B, Nkv, H, Dh).permute(0, 2, 1, 3)
        V = v_proj(kv_src).reshape(B, Nkv, H, Dh).permute(0, 2, 1, 3)
        attn = F.softmax(torch.matmul(Q, K.transpose(-2, -1)) * self._scale, dim=-1)
        out = torch.matmul(attn, V).permute(0, 2, 1, 3).reshape(B, Nq, D)
        return out_proj(out)

    def forward(self, vis_32q, implicit_8q):
        if implicit_8q is None:
            return vis_32q, implicit_8q
        attn_e = self._attn(self.q_e2i, self.k_e2i, self.v_e2i, self.out_e,
                            vis_32q.float(), implicit_8q.float())
        gate_e_val = self.gate_e(torch.cat([vis_32q.float(), attn_e], dim=-1))
        vis_fused = self.norm_e(vis_32q.float() + gate_e_val * attn_e)
        attn_i = self._attn(self.q_i2e, self.k_i2e, self.v_i2e, self.out_i,
                            implicit_8q.float(), vis_32q.float())
        gate_i_val = self.gate_i(torch.cat([implicit_8q.float(), attn_i], dim=-1))
        impl_fused = self.norm_i(implicit_8q.float() + gate_i_val * attn_i)
        return vis_fused.to(vis_32q.dtype), impl_fused.to(implicit_8q.dtype)


# ============================================================
# TriStreamFusion
# ============================================================
class TriStreamFusion(nn.Module):
    """
    三流融合:
      stage1: global_32q ← vis_32q (cross-attn + residual)
      stage2: f_roi      ← implicit_8q (cross-attn, gated residual)
    """

    def __init__(self, embed_dim=QFORMER_HIDDEN, num_heads=8):
        super().__init__()
        D = embed_dim
        Dh = D // num_heads
        self.num_heads = num_heads
        self.head_dim = Dh
        self._scale = Dh ** -0.5

        self.q1 = nn.Linear(D, D, bias=False)
        self.k1 = nn.Linear(D, D, bias=False)
        self.v1 = nn.Linear(D, D, bias=False)
        self.out1 = nn.Linear(D, D, bias=False)
        self.norm1 = nn.LayerNorm(D)

        self.q2 = nn.Linear(D, D, bias=False)
        self.k2 = nn.Linear(D, D, bias=False)
        self.v2 = nn.Linear(D, D, bias=False)
        self.out2 = nn.Linear(D, D, bias=False)
        self.norm2 = nn.LayerNorm(D)

        self.gate = nn.Sequential(
            nn.LayerNorm(D * 2), nn.Linear(D * 2, D), nn.Sigmoid(),
        )
        self.norm3 = nn.LayerNorm(D)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        n = sum(p.numel() for p in self.parameters())
        print(f"  TriStreamFusion: {n/1e6:.2f}M")

    def _xattn(self, q_proj, k_proj, v_proj, out_proj, norm,
               q_src, kv_src, residual=True):
        B, Nq, D = q_src.shape
        Nkv = kv_src.shape[1]
        H, Dh = self.num_heads, self.head_dim
        Q = q_proj(q_src.float()).reshape(B, Nq, H, Dh).permute(0, 2, 1, 3)
        K = k_proj(kv_src.float()).reshape(B, Nkv, H, Dh).permute(0, 2, 1, 3)
        V = v_proj(kv_src.float()).reshape(B, Nkv, H, Dh).permute(0, 2, 1, 3)
        attn = F.softmax(torch.matmul(Q, K.transpose(-2, -1)) * self._scale, dim=-1)
        ctx = torch.matmul(attn, V).permute(0, 2, 1, 3).reshape(B, Nq, D)
        ctx = out_proj(ctx)
        if residual:
            return norm(q_src.float() + ctx)
        return norm(ctx)

    def forward(self, global_32q, vis_32q, implicit_8q):
        f_roi = self._xattn(self.q1, self.k1, self.v1, self.out1, self.norm1,
                            global_32q, vis_32q, residual=True)
        if implicit_8q is not None:
            f_name = self._xattn(self.q2, self.k2, self.v2, self.out2, self.norm2,
                                 f_roi, implicit_8q.float(), residual=False)
            gate_val = self.gate(torch.cat([f_roi, f_name], dim=-1))
            fused = self.norm3(f_roi + gate_val * f_name)
        else:
            fused = f_roi
        return fused.to(global_32q.dtype)


# ============================================================
# GlobalROIBidirectionalFusion
# ============================================================
class GlobalROIBidirectionalFusion(nn.Module):
    """
    全局 ↔ ROI 双向融合,输出:
      [global_enriched, core_4 padded, non_core_9 padded, (spatial)]
    """

    def __init__(self, embed_dim=EVA01_DIM, num_heads=8,
                 max_core=4, max_non_core=9):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.max_core = max_core
        self.max_non_core = max_non_core

        self.importance_gate = nn.Sequential(
            nn.LayerNorm(embed_dim * 2),
            nn.Linear(embed_dim * 2, embed_dim // 4), nn.GELU(),
            nn.Linear(embed_dim // 4, 1), nn.Sigmoid(),
        )
        self.global_q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.roi_k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.roi_v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.global_out_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.global_norm = nn.LayerNorm(embed_dim)
        self.roi_norm = nn.LayerNorm(embed_dim)
        self.scale = (embed_dim // num_heads) ** -0.5
        self.residual_weight = 0.5
        for m in [self.global_q_proj, self.roi_k_proj,
                  self.roi_v_proj, self.global_out_proj]:
            nn.init.xavier_uniform_(m.weight)
        for m in self.importance_gate.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, global_feat, fused_rois, is_core,
                importance_scores, spatial_tokens=None):
        B, K, D = fused_rois.shape
        device = fused_rois.device
        g_exp = global_feat.unsqueeze(1).expand(-1, K, -1).float()
        roi_f = fused_rois.float()
        importance = self.importance_gate(torch.cat([roi_f, g_exp], dim=-1))
        weighted_rois = roi_f * importance

        H = self.num_heads
        Dh = D // H
        g_norm = self.global_norm(global_feat.float()).unsqueeze(1)
        r_norm = self.roi_norm(weighted_rois)
        q = self.global_q_proj(g_norm)
        k = self.roi_k_proj(r_norm)
        v = self.roi_v_proj(r_norm)
        q = q.reshape(B, 1, H, Dh).permute(0, 2, 1, 3)
        k = k.reshape(B, K, H, Dh).permute(0, 2, 1, 3)
        v = v.reshape(B, K, H, Dh).permute(0, 2, 1, 3)
        attn = F.softmax((q @ k.transpose(-2, -1)) * self.scale, dim=-1)
        cross_out = (attn @ v).permute(0, 2, 1, 3).reshape(B, 1, D)
        cross_out = self.global_out_proj(cross_out).squeeze(1)
        global_enriched = (global_feat.float() + self.residual_weight * cross_out
                           ).to(fused_rois.dtype)

        fusion_parts = [global_enriched.unsqueeze(1)]
        core_list, non_core_list = [], []
        for b in range(B):
            core_f = weighted_rois[b][is_core[b]][:self.max_core]
            non_f = weighted_rois[b][~is_core[b]][:self.max_non_core]
            if core_f.shape[0] < self.max_core:
                pad = torch.zeros(self.max_core - core_f.shape[0], D,
                                  device=device, dtype=fused_rois.dtype)
                core_f = torch.cat([core_f.to(fused_rois.dtype), pad])
            if non_f.shape[0] < self.max_non_core:
                pad = torch.zeros(self.max_non_core - non_f.shape[0], D,
                                  device=device, dtype=fused_rois.dtype)
                non_f = torch.cat([non_f.to(fused_rois.dtype), pad])
            core_list.append(core_f)
            non_core_list.append(non_f)
        fusion_parts.append(torch.stack(core_list))
        fusion_parts.append(torch.stack(non_core_list))
        if spatial_tokens is not None:
            fusion_parts.append(spatial_tokens[:, :self.max_core + self.max_non_core])
        return torch.cat(fusion_parts, dim=1)


# ============================================================
# ROICrossAttnRefinement
# ============================================================
class ROICrossAttnRefinement(nn.Module):
    """ROI ← patches 的 cross-attention 精炼,可选叠加空间 bias"""

    def __init__(self, embed_dim=EVA01_DIM, num_heads=8,
                 dropout=0.0, use_spatial_bias=True):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.use_spatial_bias = use_spatial_bias

        self.roi_q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.patch_k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.patch_v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.gate = nn.Sequential(
            nn.LayerNorm(embed_dim * 2),
            nn.Linear(embed_dim * 2, 1), nn.Sigmoid(),
        )
        self.norm_roi = nn.LayerNorm(embed_dim)
        self.norm_patch = nn.LayerNorm(embed_dim)
        self.drop = nn.Dropout(dropout)
        self._scale = self.head_dim ** -0.5
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)

    def _bbox_patch_overlap(self, boxes_batch, grid_h, grid_w, device):
        B = len(boxes_batch)
        N = max(len(b) for b in boxes_batch) if boxes_batch else 0
        out = torch.zeros(B, N, grid_h * grid_w, device=device)
        for b, boxes in enumerate(boxes_batch):
            for i, box in enumerate(boxes):
                if i >= N:
                    break
                x1, y1, x2, y2 = box['bbox']
                px1 = x1 * grid_w
                py1 = y1 * grid_h
                px2 = x2 * grid_w
                py2 = y2 * grid_h
                for pi in range(grid_h):
                    for pj in range(grid_w):
                        cx = pj + 0.5
                        cy = pi + 0.5
                        dx = max(0, max(px1 - cx, cx - px2))
                        dy = max(0, max(py1 - cy, cy - py2))
                        out[b, i, pi * grid_w + pj] = math.exp(
                            -(dx ** 2 + dy ** 2) ** 0.5 * 0.5)
        return out

    def forward(self, roi_feats, patches, boxes_batch=None, grid_size=16):
        B, N, D = roi_feats.shape
        P = patches.shape[1]
        H = self.num_heads
        Dh = self.head_dim
        device = roi_feats.device

        roi_n = self.norm_roi(roi_feats.float())
        patch_n = self.norm_patch(patches.float())
        Q = self.roi_q_proj(roi_n).reshape(B, N, H, Dh).permute(0, 2, 1, 3)
        K = self.patch_k_proj(patch_n).reshape(B, P, H, Dh).permute(0, 2, 1, 3)
        V = self.patch_v_proj(patch_n).reshape(B, P, H, Dh).permute(0, 2, 1, 3)
        attn = torch.matmul(Q, K.transpose(-2, -1)) * self._scale
        if self.use_spatial_bias and boxes_batch is not None:
            try:
                spatial_bias = self._bbox_patch_overlap(
                    boxes_batch, grid_size, grid_size, device)
                spatial_bias = spatial_bias[:, :N, :P].unsqueeze(1)
                attn = attn + torch.log(spatial_bias.clamp(min=1e-6)).to(attn.dtype)
            except Exception:
                pass
        attn = self.drop(F.softmax(attn, dim=-1))
        out = torch.matmul(attn, V).permute(0, 2, 1, 3).reshape(B, N, D)
        out = self.out_proj(out.float())
        gate = self.gate(torch.cat([roi_feats.float(), out], dim=-1))
        return (roi_feats.float() + gate * out).to(roi_feats.dtype)
