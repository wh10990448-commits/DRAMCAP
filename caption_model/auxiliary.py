#!/usr/bin/env python3
"""
auxiliary.py ── 辅助模块集合

包括:
  - NameAttrDeduplicator: 名称/属性语义去重
  - PKLAnchoredImplicitROI (alias IndexAnchoredImplicitROI): 名称锚定的隐式 ROI
  - SpatialTokenEmbedder: 把 abs/rel pos 转成 LLM token embedding
  - EntityTextEmbedder: 把 'tag at:[x1,y1,x2,y2]' 编码
  - VisFeatureEnhancer: 视觉特征残差增强 (跟 SigLIP 无关, 名字纯历史遗留)
  - AoAGate: Attention on Attention 门控
  - EVCapVisualProjector: 视觉到 LLM 的投影
  - GlobalContextFusion: 全局-ROI 简化门控
  - CollectiveObjectMerger: 多个相同核心物体的集体合并
  - TypeEmbedding / SpatialTagAnchor / CoreObjectAttention
  - LlipMixtureTokens: 多模式混合 tokens
  - ObjectTagEmbedder: 物体 tag → embedding
  - ContrastiveHead: 视觉-文本对比头
"""

import math
from typing import Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .constants import (
    EVA01_DIM, QFORMER_HIDDEN, LLM_HIDDEN,
    INVALID_TAG_NAMES_EXTENDED,
)
from .utils import abs_to_loc_tokens, rel_sinusoidal, evcap_pre_name, build_entity_text


# ============================================================
# NameAttrDeduplicator
# ============================================================
class NameAttrDeduplicator(nn.Module):
    """名称/属性 (LLM 文本编码) 语义去重 + token 化"""

    def __init__(self, qformer_dim=QFORMER_HIDDEN, name_sem_thresh=0.92,
                 attr_sim_thresh=0.15, max_names=8, max_attrs_per_obj=3):
        super().__init__()
        self.qformer_dim = qformer_dim
        self.name_sem_thresh = name_sem_thresh
        self.attr_sim_thresh = attr_sim_thresh
        self.max_names = max_names
        self.max_attrs_per_obj = max_attrs_per_obj
        self.name_proj = None
        self.attr_proj = None
        self._embed_fn = None
        self._tokenizer = None

    def set_embed_fn(self, embed_fn, tokenizer, llm_hidden=LLM_HIDDEN):
        self._embed_fn = embed_fn
        self._tokenizer = tokenizer
        device = next(embed_fn.parameters()).device
        if self.name_proj is None:
            self.name_proj = nn.Sequential(
                nn.Linear(llm_hidden, self.qformer_dim),
                nn.LayerNorm(self.qformer_dim), nn.GELU(),
                nn.Linear(self.qformer_dim, self.qformer_dim),
                nn.LayerNorm(self.qformer_dim),
            ).to(device)
            for m in self.name_proj.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
        if self.attr_proj is None:
            self.attr_proj = nn.Sequential(
                nn.Linear(llm_hidden, self.qformer_dim),
                nn.LayerNorm(self.qformer_dim), nn.GELU(),
                nn.Linear(self.qformer_dim, self.qformer_dim),
                nn.LayerNorm(self.qformer_dim),
            ).to(device)
            for m in self.attr_proj.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

    @torch.no_grad()
    def _encode_texts(self, texts, proj, device):
        if not self._embed_fn or not self._tokenizer:
            return torch.zeros(len(texts), self.qformer_dim, device=device)
        valid = [t if t and t.strip() else "object" for t in texts]
        enc = self._tokenizer(valid, padding=True, truncation=True, max_length=24,
                              return_tensors='pt', add_special_tokens=False).to(device)
        emb = self._embed_fn(enc['input_ids'])
        mask = enc['attention_mask'].unsqueeze(-1).float()
        pooled = (emb * mask).sum(1) / mask.sum(1).clamp(min=1)
        return proj(pooled.float())

    def deduplicate_names(self, name_list, device):
        seen = set()
        unique_names = []
        for n in name_list:
            n_clean = n.strip().lower() if n else ""
            if (n_clean and n_clean not in seen
                    and n_clean not in INVALID_TAG_NAMES_EXTENDED):
                seen.add(n_clean)
                unique_names.append(n)
        if len(unique_names) <= 1:
            return unique_names[:self.max_names]
        if self.name_proj is None:
            return unique_names[:self.max_names]
        try:
            embeds = self._encode_texts(unique_names, self.name_proj, device)
            embeds_n = F.normalize(embeds.float(), p=2, dim=-1)
            sim = embeds_n @ embeds_n.T
            keep = [True] * len(unique_names)
            for i in range(len(unique_names)):
                if not keep[i]:
                    continue
                for j in range(i + 1, len(unique_names)):
                    if keep[j] and sim[i, j].item() > self.name_sem_thresh:
                        keep[j] = False
            return [n for n, k in zip(unique_names, keep) if k][:self.max_names]
        except Exception:
            return unique_names[:self.max_names]

    def deduplicate_attrs(self, name_attr_map):
        result = {}
        for name, attrs in name_attr_map.items():
            if not attrs:
                result[name] = []
                continue
            seen_dims = {}
            for attr in sorted(attrs, key=lambda x: x.get('sim', 0), reverse=True):
                dim = attr.get('dim', 'unknown')
                if dim not in seen_dims:
                    seen_dims[dim] = attr
            filtered = [a for a in seen_dims.values()
                        if a.get('sim', 0) >= self.attr_sim_thresh]
            result[name] = [
                a['value'] for a in filtered[:self.max_attrs_per_obj]
                if a.get('value', '').strip()
                and not a['value'].strip().isdigit()
                and len(a['value'].strip()) > 1
            ]
        return result

    def encode_name_attr_tokens(self, names_batch, attrs_batch, device):
        B = len(names_batch)
        D = self.qformer_dim
        name_out = torch.zeros(B, self.max_names, D, device=device)
        attr_out = torch.zeros(B, self.max_names, D, device=device)
        name_mask = torch.zeros(B, self.max_names, dtype=torch.bool, device=device)
        for b in range(B):
            names_b = names_batch[b][:self.max_names]
            if names_b and self.name_proj is not None:
                ne = self._encode_texts(names_b, self.name_proj, device)
                name_out[b, :len(names_b)] = ne
                name_mask[b, :len(names_b)] = True
            attrs_b = attrs_batch[b] if b < len(attrs_batch) else {}
            for i, name in enumerate(names_b):
                if i >= self.max_names:
                    break
                attr_vals = attrs_b.get(name, [])
                if attr_vals and self.attr_proj is not None:
                    ae = self._encode_texts(attr_vals, self.attr_proj, device)
                    attr_out[b, i] = ae.mean(0)
        return name_out, attr_out, name_mask


# ============================================================
# PKLAnchoredImplicitROI
# ============================================================
class PKLAnchoredImplicitROI(nn.Module):
    """从 PKL 名称库 + Q-Former 输出找出隐式 ROI"""

    def __init__(self, embed_dim=QFORMER_HIDDEN, top_k_rois=8,
                 temperature=0.07, gate_threshold=0.05):
        super().__init__()
        assert embed_dim == QFORMER_HIDDEN
        self.embed_dim = embed_dim
        self.top_k_rois = top_k_rois
        self.temperature = temperature
        self.gate_threshold = gate_threshold
        self.q_proj = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim, bias=False), nn.GELU(),
        )
        self.k_proj = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim, bias=False), nn.GELU(),
        )
        self.out_proj = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim), nn.GELU(),
            nn.Linear(embed_dim, embed_dim), nn.LayerNorm(embed_dim),
        )
        self.importance_head = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim // 4), nn.GELU(),
            nn.Linear(embed_dim // 4, 1),
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        self._name_keys = None
        self._name_strings = None
        self._loaded = False

    def load_name_anchors_from_pkl(self, name_pkl_path, device, top_k=839):
        if self._loaded:
            return
        import os
        import pickle
        if not os.path.exists(name_pkl_path):
            return
        try:
            with open(name_pkl_path, 'rb') as f:
                nm = pickle.load(f)
            if isinstance(nm, dict):
                nk = nm.get('keys', nm.get('key'))
                nv = nm.get('values', nm.get('value'))
            elif isinstance(nm, (list, tuple)) and len(nm) == 2:
                nk, nv = nm
            else:
                raise ValueError(f"无法解析名称pkl: {type(nm)}")
            if isinstance(nk, np.ndarray):
                nk = torch.from_numpy(nk.astype(np.float32))
            nk = F.normalize(nk.float(), p=2, dim=-1)
            if isinstance(nv, torch.Tensor):
                nv = nv.tolist()
            if isinstance(nv, np.ndarray):
                nv = nv.tolist()
            nv_clean = [evcap_pre_name(str(v)) for v in nv]
            seen = set()
            keep_idx = []
            for i, name in enumerate(nv_clean):
                if (name and name not in seen
                        and name not in INVALID_TAG_NAMES_EXTENDED):
                    seen.add(name)
                    keep_idx.append(i)
            keep_idx = keep_idx[:top_k]
            self._name_keys = nk[keep_idx].to(device)
            self._name_strings = [nv_clean[i] for i in keep_idx]
            self._loaded = True
            print(f"  [PKLImplicitROI] ✅ {len(self._name_strings)} 名称锚")
        except Exception as e:
            print(f"  [PKLImplicitROI] ❌ 加载失败: {e}")

    def forward(self, qformer_out_768, global_feat_768=None,
                retrieval_mean_768=None, **kwargs):
        B, Q, D = qformer_out_768.shape
        device = qformer_out_768.device
        if not self._loaded or self._name_keys is None:
            dummy = torch.zeros(B, self.top_k_rois, D,
                                device=device, dtype=qformer_out_768.dtype)
            vmask = torch.zeros(B, self.top_k_rois, dtype=torch.bool, device=device)
            return dummy, [[] for _ in range(B)], vmask
        name_keys = self._name_keys.to(device=device, dtype=torch.float32)
        N = name_keys.shape[0]
        q_feat = self.q_proj(qformer_out_768.float())
        k_feat = self.k_proj(name_keys)
        q_n = F.normalize(q_feat, p=2, dim=-1)
        k_n = F.normalize(k_feat, p=2, dim=-1)
        sim = torch.bmm(q_n, k_n.T.unsqueeze(0).expand(B, -1, -1)) / self.temperature
        attn = F.softmax(sim, dim=1)
        roi_feats = torch.bmm(attn.permute(0, 2, 1), qformer_out_768.float())

        g_exp = (global_feat_768.float().unsqueeze(1).expand(-1, N, -1)
                 if global_feat_768 is not None
                 else qformer_out_768.float().mean(1).unsqueeze(1).expand(-1, N, -1))
        importance = self.importance_head(
            torch.cat([roi_feats, g_exp], dim=-1)).squeeze(-1)
        cos_sim_score = F.cosine_similarity(
            F.normalize(roi_feats, dim=-1), F.normalize(g_exp, dim=-1), dim=-1)
        attn_max = attn.max(dim=1).values
        attn_max_norm = (attn_max - attn_max.min(1, keepdim=True).values) / (
                attn_max.max(1, keepdim=True).values
                - attn_max.min(1, keepdim=True).values + 1e-8)
        final_score = importance + 0.3 * cos_sim_score + 0.2 * attn_max_norm

        gate_query = (F.normalize(retrieval_mean_768.float(), p=2, dim=-1)
                      if retrieval_mean_768 is not None
                      else F.normalize(qformer_out_768.float().mean(1), p=2, dim=-1))
        gate_scores = torch.mm(gate_query, name_keys.T)
        final_score = final_score + 0.5 * gate_scores

        k_select = min(self.top_k_rois, N)
        top_idx = final_score.topk(k_select, dim=1).indices
        selected = roi_feats.gather(1, top_idx.unsqueeze(-1).expand(-1, -1, D))
        selected = self.out_proj(selected).to(qformer_out_768.dtype)

        top_gate = gate_scores.gather(1, top_idx)
        valid_mask = top_gate > self.gate_threshold
        selected = selected * valid_mask.unsqueeze(-1).to(selected.dtype)

        top_idx_list = top_idx.cpu().tolist()
        valid_list = valid_mask.cpu().tolist()
        top_names = [
            [self._name_strings[top_idx_list[b][j]]
             if (j < len(valid_list[b]) and valid_list[b][j]
                 and top_idx_list[b][j] < len(self._name_strings)) else ""
             for j in range(k_select)]
            for b in range(B)
        ]
        return selected, top_names, valid_mask

    @property
    def num_names(self):
        return len(self._name_strings) if self._name_strings else 0


# 兼容别名
IndexAnchoredImplicitROI = PKLAnchoredImplicitROI


# ============================================================
# SpatialTokenEmbedder
# ============================================================
class SpatialTokenEmbedder(nn.Module):
    """把 abs/rel 位置信息转换成 LLM 词嵌入空间"""

    def __init__(self, embed_fn, tokenizer, llm_hidden=LLM_HIDDEN,
                 out_dim=EVA01_DIM, sin_dim=32):
        super().__init__()
        self.embed_fn = embed_fn
        self.tokenizer = tokenizer
        self.sin_dim = sin_dim
        rel_sin_dim = 12 * sin_dim
        self.abs_proj = nn.Sequential(
            nn.Linear(llm_hidden, out_dim), nn.LayerNorm(out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim), nn.LayerNorm(out_dim),
        )
        self.rel_mlp = nn.Sequential(
            nn.Linear(rel_sin_dim, out_dim), nn.LayerNorm(out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim), nn.LayerNorm(out_dim),
        )

    @torch.no_grad()
    def _bin_tokens_to_embeds(self, texts):
        device = next(self.embed_fn.parameters()).device
        enc = self.tokenizer(texts, padding=True, truncation=True, max_length=32,
                             return_tensors='pt', add_special_tokens=False).to(device)
        embeds = self.embed_fn(enc['input_ids'])
        mask = enc['attention_mask'].unsqueeze(-1).float()
        return (embeds * mask).sum(1) / mask.sum(1).clamp(min=1)

    def encode_abs(self, abs_pos):
        B, N, _ = abs_pos.shape
        texts = [abs_to_loc_tokens(abs_pos.reshape(B * N, 10)[i].cpu())
                 for i in range(B * N)]
        return self.abs_proj(
            self._bin_tokens_to_embeds(texts).float()
        ).reshape(B, N, -1)

    def encode_rel(self, rel_pos):
        B, N, _, _ = rel_pos.shape
        device = rel_pos.device
        if N > 1:
            rel_sin_list = []
            for b in range(B):
                obj_vecs = [
                    torch.stack([
                        rel_sinusoidal(rel_pos[b, i, j], self.sin_dim)
                        for j in range(N) if j != i
                    ]).mean(0)
                    for i in range(N)
                ]
                rel_sin_list.append(torch.stack(obj_vecs))
            rel_sin = torch.stack(rel_sin_list)
        else:
            rel_sin = torch.zeros(B, N, 12 * self.sin_dim,
                                  dtype=torch.float, device=device)
        return self.rel_mlp(rel_sin.float())


# ============================================================
# EntityTextEmbedder
# ============================================================
class EntityTextEmbedder(nn.Module):
    """编码 'tag at:[x1,y1,x2,y2]' 字符串"""

    def __init__(self, embed_fn, tokenizer, llm_hidden=LLM_HIDDEN, out_dim=EVA01_DIM):
        super().__init__()
        self.embed_fn = embed_fn
        self.tokenizer = tokenizer
        self.entity_proj = nn.Sequential(
            nn.Linear(llm_hidden, out_dim), nn.LayerNorm(out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim), nn.LayerNorm(out_dim),
        )

    @torch.no_grad()
    def _texts_to_pool(self, texts):
        device = next(self.embed_fn.parameters()).device
        enc = self.tokenizer(texts, padding=True, truncation=True, max_length=40,
                             return_tensors='pt', add_special_tokens=False).to(device)
        embeds = self.embed_fn(enc['input_ids'])
        mask = enc['attention_mask'].unsqueeze(-1).float()
        return (embeds * mask).sum(1) / mask.sum(1).clamp(min=1)

    def encode(self, tag_sequences, abs_pos):
        B, N, _ = abs_pos.shape
        texts = [
            build_entity_text(
                tag_sequences[b][n] if (
                    tag_sequences and b < len(tag_sequences)
                    and n < len(tag_sequences[b])
                ) else "object",
                abs_pos[b, n].cpu(),
            )
            for b in range(B) for n in range(N)
        ]
        return self.entity_proj(
            self._texts_to_pool(texts).float()
        ).reshape(B, N, -1)


# ============================================================
# VisFeatureEnhancer  (原名 SigLIPNativeAligner, 跟 SigLIP 无关)
# ============================================================
class VisFeatureEnhancer(nn.Module):
    """视觉特征轻量残差增强: out = x + scale * MLP(LN(x))

    历史名 SigLIPNativeAligner 是误导性命名 ── 内部就是一个通用
    残差 MLP, 与 SigLIP encoder 没有任何关系.
    """

    def __init__(self, embed_dim=EVA01_DIM):
        super().__init__()
        self.enhancer = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim), nn.GELU(),
            nn.Linear(embed_dim, embed_dim), nn.LayerNorm(embed_dim),
        )
        self.scale = nn.Parameter(torch.ones(1))

    def forward(self, x):
        return x + self.scale * self.enhancer(x)

    def forward_for_loss(self, x):
        return F.normalize(self.forward(x), p=2, dim=-1)


# ============================================================
# AoAGate
# ============================================================
class AoAGate(nn.Module):
    """Attention on Attention 门控"""

    def __init__(self, embed_dim=QFORMER_HIDDEN):
        super().__init__()
        D = embed_dim
        self.gate_proj = nn.Linear(D * 2, D)
        self.info_proj = nn.Linear(D * 2, D)
        self.out_norm = nn.LayerNorm(D)
        nn.init.xavier_uniform_(self.gate_proj.weight)
        nn.init.zeros_(self.gate_proj.bias)
        nn.init.xavier_uniform_(self.info_proj.weight)
        nn.init.zeros_(self.info_proj.bias)

    def forward(self, attn_out, query_in):
        cat = torch.cat([attn_out.float(), query_in.float()], dim=-1)
        gate = torch.sigmoid(self.gate_proj(cat))
        info = torch.tanh(self.info_proj(cat))
        return (attn_out.float() + self.out_norm(gate * info)).to(attn_out.dtype)


# ============================================================
# EVCapVisualProjector
# ============================================================
class EVCapVisualProjector(nn.Module):
    """视觉 → LLM 的投影"""

    def __init__(self, vis_dim=QFORMER_HIDDEN, llm_hidden=LLM_HIDDEN, mid_dim=2048):
        super().__init__()
        self.projector = nn.Sequential(
            nn.Linear(vis_dim, mid_dim), nn.GELU(),
            nn.Linear(mid_dim, llm_hidden), nn.LayerNorm(llm_hidden),
        )
        for m in self.projector.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.8)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.projector(x)


# ============================================================
# GlobalContextFusion
# ============================================================
class GlobalContextFusion(nn.Module):
    """轻量 global → ROI 门控"""

    def __init__(self, embed_dim=EVA01_DIM):
        super().__init__()
        self.gate_mlp = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim), nn.Sigmoid(),
        )
        for m in self.gate_mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, roi_feat, global_feat):
        return roi_feat * (1.0 + 0.5 * self.gate_mlp(global_feat).unsqueeze(1))


# ============================================================
# CollectiveObjectMerger
# ============================================================
class CollectiveObjectMerger(nn.Module):
    """同类核心物体≥3个时,合并成 1 个 (加 collective_bias)"""

    def __init__(self, embed_dim=EVA01_DIM, merge_threshold=3):
        super().__init__()
        self.merge_threshold = merge_threshold
        self.collective_bias = nn.Parameter(torch.zeros(embed_dim))
        nn.init.normal_(self.collective_bias, std=0.01)

    def merge(self, roi_feats, is_core, category_ids, importance_scores):
        from collections import defaultdict
        N = roi_feats.shape[0]
        device = roi_feats.device
        category_ids = list(category_ids)
        if len(category_ids) < N:
            category_ids += [0] * (N - len(category_ids))
        core_by_cat = defaultdict(list)
        for i in range(N):
            if is_core[i]:
                core_by_cat[category_ids[i]].append(i)
        to_merge = {c: idxs for c, idxs in core_by_cat.items()
                    if len(idxs) >= self.merge_threshold}
        if not to_merge:
            return roi_feats, is_core, category_ids, importance_scores, {}
        skip = set(i for idxs in to_merge.values() for i in idxs)
        mf, mc, mcat, ms, mi = [], [], [], [], {}
        midx = 0
        for i in range(N):
            if i in skip:
                continue
            mf.append(roi_feats[i])
            mc.append(is_core[i].item())
            mcat.append(category_ids[i])
            ms.append(importance_scores[i].item())
            midx += 1
        for cat, idxs in to_merge.items():
            w = importance_scores[idxs] / (importance_scores[idxs].sum() + 1e-8)
            mf.append((roi_feats[idxs] * w.unsqueeze(-1)).sum(0)
                      + self.collective_bias)
            mc.append(True)
            mcat.append(cat)
            ms.append(float(importance_scores[idxs].max()))
            mi[tuple(idxs)] = midx
            midx += 1
        return (torch.stack(mf),
                torch.tensor(mc, dtype=torch.bool, device=device),
                mcat,
                torch.tensor(ms, dtype=torch.float32, device=device),
                mi)


# ============================================================
# TypeEmbedding
# ============================================================
class TypeEmbedding(nn.Module):
    def __init__(self, num_types=8, embed_dim=EVA01_DIM):
        super().__init__()
        self.emb = nn.Embedding(num_types, embed_dim)

    def forward(self, t):
        return self.emb(t)


# ============================================================
# CoreObjectAttention
# ============================================================
class CoreObjectAttention(nn.Module):
    """核心物体加 core_embed bias + importance 缩放"""

    def __init__(self, embed_dim=EVA01_DIM):
        super().__init__()
        self.core_embed = nn.Parameter(torch.zeros(embed_dim))
        nn.init.normal_(self.core_embed, std=0.01)
        self.importance_proj = nn.Sequential(
            nn.Linear(1, embed_dim // 4), nn.ReLU(),
            nn.Linear(embed_dim // 4, 1), nn.Sigmoid(),
        )
        for m in self.importance_proj.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, roi_features, is_core=None, importance_scores=None):
        out = roi_features
        if is_core is not None:
            out = out + is_core.float().unsqueeze(-1) * self.core_embed
        if importance_scores is not None:
            scale = self.importance_proj(importance_scores.unsqueeze(-1).float())
            out = out * (1.0 + 0.4 * scale)
        return out, None


# ============================================================
# SpatialTagAnchor
# ============================================================
class SpatialTagAnchor(nn.Module):
    """attr / abs / rel / entity 4 个空间锚点的 gated 融合"""

    def __init__(self, embed_dim=EVA01_DIM, abs_dim=10, rel_dim=12):
        super().__init__()
        D = embed_dim
        H = D // 2
        self.shared_gate = nn.Sequential(
            nn.Linear(D * 2, D // 4), nn.GELU(),
            nn.Linear(D // 4, 1), nn.Sigmoid(),
        )

        def _proj():
            return nn.Sequential(
                nn.Linear(D * 2, H), nn.LayerNorm(H), nn.GELU(),
                nn.Linear(H, D), nn.LayerNorm(D),
            )

        self.attr_proj = _proj()
        self.abs_proj_fuse = _proj()
        self.rel_proj_fuse = _proj()
        self.entity_proj_fuse = _proj()
        self.spatial_embedder = None
        self.entity_embedder = None
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def set_embedder(self, e):
        self.spatial_embedder = e

    def set_entity_embedder(self, e):
        self.entity_embedder = e

    def forward(self, roi_feat, attr_feat, abs_pos, rel_pos,
                tag_sequences=None, is_core=None):
        B, N, D = roi_feat.shape
        device = roi_feat.device
        out = roi_feat
        core_mask = (is_core.float().unsqueeze(-1).to(roi_feat.dtype)
                     if is_core is not None
                     else torch.ones(B, N, 1, device=device, dtype=roi_feat.dtype))
        c1 = torch.cat([out, attr_feat.to(roi_feat.dtype)], dim=-1)
        out = out + self.shared_gate(c1) * self.attr_proj(c1)

        abs_f = (self.spatial_embedder.encode_abs(abs_pos).to(roi_feat.dtype)
                 if self.spatial_embedder
                 else torch.zeros(B, N, D, device=device, dtype=roi_feat.dtype))
        c2 = torch.cat([out, abs_f], dim=-1)
        out = out + core_mask * self.shared_gate(c2) * self.abs_proj_fuse(c2)

        rel_f = (self.spatial_embedder.encode_rel(rel_pos).to(roi_feat.dtype)
                 if self.spatial_embedder
                 else torch.zeros(B, N, D, device=device, dtype=roi_feat.dtype))
        c3 = torch.cat([out, rel_f], dim=-1)
        out = out + core_mask * self.shared_gate(c3) * self.rel_proj_fuse(c3)

        if self.entity_embedder is not None and tag_sequences is not None:
            ent_f = self.entity_embedder.encode(tag_sequences, abs_pos
                                                ).to(roi_feat.dtype)
            c4 = torch.cat([out, ent_f], dim=-1)
            out = out + core_mask * self.shared_gate(c4) * self.entity_proj_fuse(c4)
        return out


# ============================================================
# LlipMixtureTokens
# ============================================================
class LlipMixtureTokens(nn.Module):
    """多个 mixture queries 经 transformer encode 得到多模式表征"""

    def __init__(self, embed_dim=EVA01_DIM, num_mixture=4,
                 num_heads=8, num_layers=2):
        super().__init__()
        self.num_mixture = num_mixture
        self.mix_tokens = nn.Parameter(
            torch.randn(1, num_mixture, embed_dim) * 0.02)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embed_dim, nhead=num_heads,
                dim_feedforward=embed_dim * 4, dropout=0.1,
                batch_first=True, activation='gelu',
            ),
            num_layers=num_layers,
        )
        self.out_norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        B = x.shape[0]
        combined = torch.cat([self.mix_tokens.expand(B, -1, -1), x], dim=1)
        return self.out_norm(self.transformer(combined)[:, :self.num_mixture])

    def forward_for_contrast(self, x):
        out = self.forward(x)
        return out, F.normalize(out.mean(1), p=2, dim=-1)


# ============================================================
# ObjectTagEmbedder
# ============================================================
class ObjectTagEmbedder(nn.Module):
    """物体 tag 字符串 → embedding (用于构造 attr_feats)"""

    def __init__(self, embed_fn, tokenizer, llm_hidden=LLM_HIDDEN, out_dim=EVA01_DIM):
        super().__init__()
        self.embed_fn = embed_fn
        self.tokenizer = tokenizer
        self.tag_proj = nn.Sequential(
            nn.Linear(llm_hidden, out_dim), nn.LayerNorm(out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim), nn.LayerNorm(out_dim),
        )
        for m in self.tag_proj.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    @torch.no_grad()
    def _text_to_embed(self, text, device):
        ids = self.tokenizer.encode(
            text, add_special_tokens=False, return_tensors='pt').to(device)
        return self.embed_fn(ids).mean(dim=1).squeeze(0)

    def encode_batch(self, tag_name_strs, device):
        embeds = [
            self._text_to_embed(t, device) if t and t.strip()
            else torch.zeros(self.tag_proj[0].in_features, device=device)
            for t in tag_name_strs
        ]
        return self.tag_proj(torch.stack(embeds).float()).unsqueeze(1)


# ============================================================
# ContrastiveHead
# ============================================================
class ContrastiveHead(nn.Module):
    """视觉 32q 与文本 caption 的对比损失"""

    def __init__(self, vis_dim=QFORMER_HIDDEN, llm_hidden=LLM_HIDDEN):
        super().__init__()
        self.text_head = nn.Linear(llm_hidden, vis_dim, bias=False)
        nn.init.xavier_uniform_(self.text_head.weight)
        self.log_temp = nn.Parameter(torch.tensor(2.6593))
        print(f"  ContrastiveHead: text_head Linear({llm_hidden}→{vis_dim}) "
              f"+ learnable temperature")

    @property
    def temperature(self):
        return torch.clamp(self.log_temp.exp(), min=0.01, max=10.0)

    def forward(self, vis_32q, cap_ids, llm_embed_fn, pad_id=2):
        B = vis_32q.shape[0]
        device = vis_32q.device
        vis_vec = F.normalize(vis_32q.float().mean(dim=1), p=2, dim=-1)
        with torch.no_grad():
            txt_emb = llm_embed_fn(cap_ids.to(device))
        mask = (cap_ids != pad_id).unsqueeze(-1).float().to(device)
        txt_mean = (txt_emb.float() * mask).sum(1) / mask.sum(1).clamp(min=1)
        txt_vec = F.normalize(self.text_head(txt_mean), p=2, dim=-1)
        tau = self.temperature
        sim = torch.matmul(vis_vec, txt_vec.T) / tau
        labels = torch.arange(B, device=device)
        loss_i2t = F.cross_entropy(sim, labels)
        loss_t2i = F.cross_entropy(sim.T, labels)
        return (loss_i2t + loss_t2i) / 2.0