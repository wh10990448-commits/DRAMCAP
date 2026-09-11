#!/usr/bin/env python3
"""
retrieval_qformer.py ── retrieval / attribution 用 Q-Former
                        (基于 EVCap/LAVIS 官方 BertLMHeadModel)

[REFACTOR-v6.4]
  - LAVISQFormerForRetrieval: 也直接用 lavis_qformer.BertLMHeadModel
    (跟 BLIP2QFormer 同一份, 只是用法上是 frozen retrieval branch)
  - RetrievalQFormer: wrapper, 从 BLIP2 ckpt 加载 frozen 权重, 提供归一化检索向量
  - AttrQFormer (新): 加上 attribution 的 q-former — 也用 BertLMHeadModel,
    专门给 attr 通路用 (与 image Q-Former 完全相同结构, 但独立训练/初始化,
    用 PKL attr_keys 做 init query 跑 cross-attention 精炼).

EVCap 的对应关系:
  - 我们的 LAVISQFormerForRetrieval ≈ EVCap 的 self.Qformer (frozen image branch)
  - 我们的 BLIP2QFormer (qformer.py) 也 ≈ EVCap 的 self.Qformer (trainable image branch)
  - 我们的 AttrQFormer (新) 是为 attr 路径专门复制的一份 image branch Q-Former
"""

import os
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from .constants import QFORMER_HIDDEN, EVA01_DIM
from .utils import patch_torch_load_safety
from . import lavis_qformer as _lavis  # EVCap 原版 BertLMHeadModel
from .qformer import _build_lavis_qformer, _strip_qformer_for_blip2

logging.getLogger("transformers").setLevel(logging.ERROR)


# ============================================================
# LAVISQFormerForRetrieval
# ============================================================
class LAVISQFormerForRetrieval(nn.Module):
    """
    Frozen image-branch Q-Former, 专门跑检索向量.
    内部用 EVCap/LAVIS 的 BertLMHeadModel — 与 BLIP2QFormer 同根同源,
    但独立实例 + 独立 ckpt 加载 + 永久 frozen.
    """
    def __init__(self, num_query_token=32, vision_width=EVA01_DIM,
                 hidden_size=QFORMER_HIDDEN, num_hidden_layers=12,
                 num_attention_heads=12, intermediate_size=3072,
                 cross_attention_freq=2):
        super().__init__()
        self.num_query_token = num_query_token
        self.hidden_size = hidden_size
        self.vision_width = vision_width

        self.bert_model, self.query_tokens = _build_lavis_qformer(
            num_query_token=num_query_token,
            vision_width=vision_width,
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            intermediate_size=intermediate_size,
            cross_attention_freq=cross_attention_freq,
        )
        self.bert_model = _strip_qformer_for_blip2(self.bert_model)
        self.ln_vision = nn.LayerNorm(vision_width)

        self._loaded = False
        print(f"  [LAVISQFormerForRetrieval] (LAVIS) {num_hidden_layers}层, {num_query_token}q")

    def load_blip2_weights(self, checkpoint_path):
        if not os.path.exists(checkpoint_path):
            print(f"  [LAVISQFormerForRetrieval] ⚠️  权重不存在: {checkpoint_path}")
            return
        patch_torch_load_safety()
        ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        state = ckpt.get('model', ckpt) if isinstance(ckpt, dict) else ckpt

        # query_tokens
        if 'query_tokens' in state:
            qt = state['query_tokens']
            if qt.shape == self.query_tokens.shape:
                self.query_tokens.data.copy_(qt)
                print(f"  [LAVISQFormerForRetrieval] ✅ query_tokens 加载: {tuple(qt.shape)}")
            elif qt.shape[-1] == self.hidden_size:
                n_pre = qt.shape[1]
                if n_pre != self.num_query_token:
                    qt = F.interpolate(
                        qt.permute(0, 2, 1).float(),
                        size=self.num_query_token,
                        mode='linear', align_corners=True
                    ).permute(0, 2, 1)
                self.query_tokens.data.copy_(qt.to(self.query_tokens.dtype))
                print(f"  [LAVISQFormerForRetrieval] ✅ query_tokens 插值: {n_pre}→{self.num_query_token}")

        # ln_vision
        for key in ('Qformer.ln_vision', 'ln_vision'):
            if f'{key}.weight' in state:
                w = state[f'{key}.weight']
                if w.shape == self.ln_vision.weight.shape:
                    self.ln_vision.weight.data.copy_(w)
                    if f'{key}.bias' in state:
                        self.ln_vision.bias.data.copy_(state[f'{key}.bias'])
                    print(f"  [LAVISQFormerForRetrieval] ✅ ln_vision 加载")
                break

        # Qformer.bert.* → bert_model.*
        bert_sd = {k[len('Qformer.'):]: v for k, v in state.items()
                   if k.startswith('Qformer.')}
        if bert_sd:
            try:
                msg = self.bert_model.load_state_dict(bert_sd, strict=False)
                missing = len(getattr(msg, 'missing_keys', []))
                unexpected = len(getattr(msg, 'unexpected_keys', []))
                own = dict(self.bert_model.named_parameters())
                n_loaded = sum(
                    own[k].numel() for k in bert_sd
                    if k in own and own[k].shape == bert_sd[k].shape)
                print(f"  [LAVISQFormerForRetrieval] ✅ 层权重: ≈{n_loaded/1e6:.2f}M  "
                      f"missing={missing} unexpected={unexpected}")
            except Exception as e:
                print(f"  [LAVISQFormerForRetrieval] ⚠️  state_dict 加载失败: {e}")

        # 永久 freeze + eval
        for p in self.parameters():
            p.requires_grad = False
        self.eval()
        self._loaded = True

    def forward(self, eva01_patch_1408, visual_mask=None):
        B = eva01_patch_1408.shape[0]
        device = eva01_patch_1408.device

        vis = torch.nan_to_num(
            self.ln_vision(eva01_patch_1408.float()),
            nan=0.0, posinf=1e4, neginf=-1e4)
        if visual_mask is None:
            visual_mask = torch.ones(B, vis.shape[1], dtype=torch.long, device=device)
        elif visual_mask.dtype != torch.long:
            visual_mask = visual_mask.long()

        queries = self.query_tokens.expand(B, -1, -1).float()
        outputs = self.bert_model.bert(
            query_embeds=queries,
            encoder_hidden_states=vis,
            encoder_attention_mask=visual_mask,
            return_dict=True,
        )
        return outputs.last_hidden_state


# ============================================================
# RetrievalQFormer (wrapper)
# ============================================================
class RetrievalQFormer(nn.Module):
    """
    上层 wrapper. 行为优先级:
      1. 如果 self.lavis_qformer 已加载预训练 → 用它输出 frozen retrieval feats
      2. 否则 fallback 到外部传入的 self.qformer (例如 model.evcap_retrieval_qformer)
      3. 否则 fallback 到零向量
    """
    def __init__(self, num_queries=32):
        super().__init__()
        self.num_queries = num_queries
        self.lavis_qformer = None  # 类型: Optional[LAVISQFormerForRetrieval]
        self.qformer = None        # 类型: Optional[BLIP2QFormer]
        self.post_norm = nn.LayerNorm(QFORMER_HIDDEN)
        nn.init.ones_(self.post_norm.weight)
        nn.init.zeros_(self.post_norm.bias)
        for p in self.post_norm.parameters():
            p.requires_grad = False
        self._loaded = False

    def set_qformer(self, qformer_module):
        self.qformer = qformer_module

    def load_blip2_weights(self, checkpoint_path):
        self.lavis_qformer = LAVISQFormerForRetrieval(
            num_query_token=self.num_queries, vision_width=EVA01_DIM,
            hidden_size=QFORMER_HIDDEN, num_hidden_layers=12,
            num_attention_heads=12, intermediate_size=3072, cross_attention_freq=2)
        self.lavis_qformer.load_blip2_weights(checkpoint_path)
        self._loaded = self.lavis_qformer._loaded
        if self._loaded:
            print(f"  [RetrievalQFormer] ✅ LAVIS Q-Former 就绪")

    def _to_device(self, device):
        if self.lavis_qformer is not None:
            cur_dev = next(self.lavis_qformer.parameters()).device
            if cur_dev != device:
                self.lavis_qformer = self.lavis_qformer.to(device)

    def forward(self, eva01_patch_1408):
        B = eva01_patch_1408.shape[0]
        device = eva01_patch_1408.device

        # 优先级 1: 用 lavis_qformer
        if self.lavis_qformer is not None and self._loaded:
            self._to_device(device)
            with torch.no_grad():
                compressed = self.lavis_qformer(eva01_patch_1408.float())
            retrieval_vecs = F.normalize(compressed.float(), p=2, dim=-1)
            return compressed.to(eva01_patch_1408.dtype), retrieval_vecs

        # 优先级 2: fallback 到外部 qformer (例如 model.evcap_retrieval_qformer)
        if self.qformer is not None:
            try:
                compressed = self.qformer.forward_raw(eva01_patch_1408)
                retrieval_vecs = F.normalize(compressed.float(), p=2, dim=-1)
                return compressed.to(eva01_patch_1408.dtype), retrieval_vecs
            except Exception:
                pass

        # 优先级 3: 零向量
        z = torch.zeros(B, self.num_queries, QFORMER_HIDDEN,
                        device=device, dtype=eva01_patch_1408.dtype)
        return z, F.normalize(z.float(), p=2, dim=-1)


# ============================================================
# [新增] AttrQFormer ── attribution 专用 Q-Former
#   设计: 与 BLIP2QFormer 完全相同结构 (LAVIS BertLMHeadModel),
#        独立实例化, 独立训练, 跟 image Q-Former 共享 BLIP-2 初始化
#        但独立解耦, 专门跑 attr 通路.
# ============================================================
class AttrQFormer(nn.Module):
    """
    Attribution Q-Former.

    使用场景:
      - 把 PKL 检索到的 attr embedding (768维) 当 init query
      - cross-attention 进图像 patch (1408维) 做精炼
      - 输出过 output_proj 给 LLM 用作 attr_tokens

    与 BLIP2QFormer 的差别:
      - 完全独立的实例 (不共享 query_tokens / bert_model 权重)
      - 默认更少 query (16 个, 跟 v6.3 的 evcap_qformer 数量对齐)
      - 但都从同一个 BLIP-2 ckpt 初始化 (load_pretrained 支持)
    """
    def __init__(self, num_queries=16, num_layers=12, num_heads=12,
                 intermediate_size=3072, cross_attention_freq=2):
        super().__init__()
        self.num_queries = num_queries
        self.hidden_size = QFORMER_HIDDEN
        self.vis_dim = EVA01_DIM

        self.bert_model, self.query_tokens = _build_lavis_qformer(
            num_query_token=num_queries,
            vision_width=self.vis_dim,
            hidden_size=self.hidden_size,
            num_hidden_layers=num_layers,
            num_attention_heads=num_heads,
            intermediate_size=intermediate_size,
            cross_attention_freq=cross_attention_freq,
        )
        self.bert_model = _strip_qformer_for_blip2(self.bert_model)
        self.ln_vision = nn.LayerNorm(self.vis_dim)

        D = self.hidden_size
        self.norm = nn.LayerNorm(D)
        self.output_proj = nn.Sequential(nn.Linear(D, D), nn.LayerNorm(D))
        nn.init.xavier_uniform_(self.output_proj[0].weight)
        nn.init.zeros_(self.output_proj[0].bias)

        # 外部 init query 投影 (PKL attr embedding → query 子空间)
        self.query_in_proj = nn.Linear(D, D, bias=True)
        nn.init.xavier_uniform_(self.query_in_proj.weight)
        nn.init.zeros_(self.query_in_proj.bias)

        ca_layers = num_layers // cross_attention_freq
        print(f"  AttrQFormer (LAVIS): {num_layers}层, {num_queries}q, "
              f"cross_attn={ca_layers}层")

    @property
    def encoder(self):
        bert_encoder = self.bert_model.bert.encoder
        if not hasattr(bert_encoder, "layers"):
            bert_encoder.layers = bert_encoder.layer
        for L in bert_encoder.layer:
            if not hasattr(L, "has_cross_attn"):
                L.has_cross_attn = getattr(L, "has_cross_attention", False)
            if not hasattr(L, "cross_attn") and hasattr(L, "crossattention"):
                L.cross_attn = L.crossattention
        return bert_encoder

    def load_pretrained(self, checkpoint_path):
        """从 BLIP-2 ckpt 加载 (跟 BLIP2QFormer.load_pretrained 同一逻辑)"""
        if not os.path.exists(checkpoint_path):
            print(f"  ⚠️  AttrQFormer 权重不存在: {checkpoint_path}")
            return
        patch_torch_load_safety()
        ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        state_dict = ckpt.get('model', ckpt)

        if 'query_tokens' in state_dict:
            qt = state_dict['query_tokens']
            if qt.shape == self.query_tokens.shape:
                self.query_tokens.data.copy_(qt)
            elif qt.shape[-1] == self.hidden_size:
                n_pre = qt.shape[1]
                if n_pre != self.num_queries:
                    qt = F.interpolate(
                        qt.permute(0, 2, 1).float(),
                        size=self.num_queries, mode='linear', align_corners=True
                    ).permute(0, 2, 1)
                self.query_tokens.data.copy_(qt.to(self.query_tokens.dtype))

        for key in ('Qformer.ln_vision', 'ln_vision'):
            if f'{key}.weight' in state_dict:
                w = state_dict[f'{key}.weight']
                if w.shape == self.ln_vision.weight.shape:
                    self.ln_vision.weight.data.copy_(w)
                    if f'{key}.bias' in state_dict:
                        self.ln_vision.bias.data.copy_(state_dict[f'{key}.bias'])
                break

        bert_sd = {k[len('Qformer.'):]: v for k, v in state_dict.items()
                   if k.startswith('Qformer.')}
        if bert_sd:
            try:
                self.bert_model.load_state_dict(bert_sd, strict=False)
                own = dict(self.bert_model.named_parameters())
                n_loaded = sum(
                    own[k].numel() for k in bert_sd
                    if k in own and own[k].shape == bert_sd[k].shape)
                print(f"  ✅ AttrQFormer (LAVIS): bert_params≈{n_loaded/1e6:.2f}M")
            except Exception as e:
                print(f"  ⚠️  AttrQFormer state_dict 加载部分失败: {e}")

    def freeze_query_and_kv_only(self):
        for p in self.parameters():
            p.requires_grad = False
        self.query_tokens.requires_grad = True
        for layer in self.bert_model.bert.encoder.layer:
            if getattr(layer, "has_cross_attention", False):
                for p in layer.crossattention.self.key.parameters():
                    p.requires_grad = True
                for p in layer.crossattention.self.value.parameters():
                    p.requires_grad = True
        for p in self.output_proj.parameters():
            p.requires_grad = True
        for p in self.query_in_proj.parameters():
            p.requires_grad = True

    freeze_bert = freeze_query_and_kv_only

    # ── forward ──
    def _run_bert(self, init_query, vis_1408, vis_mask=None):
        B = vis_1408.shape[0]
        device = vis_1408.device
        vis = torch.nan_to_num(
            self.ln_vision(vis_1408.float()),
            nan=0.0, posinf=1e4, neginf=-1e4)
        if vis_mask is None:
            vis_mask = torch.ones(B, vis.shape[1], dtype=torch.long, device=device)
        elif vis_mask.dtype != torch.long:
            vis_mask = vis_mask.long()
        outputs = self.bert_model.bert(
            query_embeds=init_query.float(),
            encoder_hidden_states=vis,
            encoder_attention_mask=vis_mask,
            return_dict=True,
        )
        return outputs.last_hidden_state

    def forward(self, visual_features_1408, visual_mask=None):
        """用自带 query_tokens 跑 attr 通路 (输出过 output_proj)"""
        B = visual_features_1408.shape[0]
        queries = self.query_tokens.expand(B, -1, -1)
        out = self._run_bert(queries, visual_features_1408, visual_mask)
        out = self.norm(out)
        return self.output_proj(out)

    def forward_with_queries(self, init_queries_768, visual_features_1408, visual_mask=None):
        """[关键] 用 PKL attr embedding 当 init queries"""
        queries = self.query_in_proj(init_queries_768.float())
        out = self._run_bert(queries, visual_features_1408, visual_mask)
        out = self.norm(out)
        return self.output_proj(out)