#!/usr/bin/env python3
"""
qformer.py ── BLIP2QFormer (基于 EVCap/LAVIS 官方 BertLMHeadModel)

[REFACTOR-v6.4]
  - 抛弃手写复现,直接用 EVCap 上传的 lavis_qformer.BertLMHeadModel
  - 与 EVCap 完全一致: cls=None, embeddings.word_embeddings=None,
    embeddings.position_embeddings=None, layer.output=None,
    layer.intermediate=None (text 端 FFN 删掉, 只留 query 端 FFN)
  - 直接吃 LAVIS blip2_pretrained.pth 的 state_dict (load_pretrained)

对外接口保持不变:
  - num_queries, hidden_size, vis_dim
  - load_pretrained(checkpoint_path)
  - freeze_query_and_kv_only() / freeze_bert
  - forward(visual_features_1408, visual_mask=None)        → 32 queries (输出过 output_proj)
  - forward_raw(visual_features_1408, visual_mask=None)    → 32 queries (不过 output_proj)
  - forward_with_queries(init_q_768, vis_1408, vis_mask)   → K queries init from external embeddings
"""

import os
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from .constants import QFORMER_HIDDEN, EVA01_DIM
from .utils import patch_torch_load_safety
from . import lavis_qformer as _lavis  # EVCap 原版 BertLMHeadModel

logging.getLogger("transformers").setLevel(logging.ERROR)


# ============================================================
# 工厂: 按 EVCap 的方式创建 LAVIS Qformer (image branch)
# ============================================================
def _build_lavis_qformer(num_query_token: int,
                         vision_width: int,
                         hidden_size: int = 768,
                         num_hidden_layers: int = 12,
                         num_attention_heads: int = 12,
                         intermediate_size: int = 3072,
                         cross_attention_freq: int = 2):
    """
    完全对齐 EVCap blip2.py 的 init_Qformer:
        encoder_config = BertConfig.from_pretrained("bert-base-uncased")
        encoder_config.encoder_width = vision_width
        encoder_config.add_cross_attention = True
        encoder_config.cross_attention_freq = cross_attention_freq
        encoder_config.query_length = num_query_token
        Qformer = BertLMHeadModel(config=encoder_config)
        query_tokens = nn.Parameter(torch.zeros(1, num_query_token, hidden_size))
        query_tokens.data.normal_(mean=0.0, std=encoder_config.initializer_range)

    返回 (qformer, query_tokens). 然后调用方做 EVCap 的去权重处理:
        qformer.cls = None
        qformer.bert.embeddings.word_embeddings = None
        qformer.bert.embeddings.position_embeddings = None
        for layer in qformer.bert.encoder.layer:
            layer.output = None
            layer.intermediate = None
    """
    # 按 EVCap 原版,优先尝试从 bert-base-uncased 加载默认 config
    # (本地无网络时可能失败,fallback 到手写 config)
    try:
        encoder_config = _lavis.BertConfig.from_pretrained("bert-base-uncased")
    except Exception:
        encoder_config = _lavis.BertConfig(
            vocab_size=30523,
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            intermediate_size=intermediate_size,
            hidden_act="gelu",
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
            max_position_embeddings=512,
            type_vocab_size=2,
            initializer_range=0.02,
            layer_norm_eps=1e-12,
            pad_token_id=0,
        )
    # ★ 用调用者参数覆盖默认值 (BLIP-2 默认就是 12/12/3072, 但允许小模型/调试用更少层)
    encoder_config.hidden_size         = hidden_size
    encoder_config.num_hidden_layers   = num_hidden_layers
    encoder_config.num_attention_heads = num_attention_heads
    encoder_config.intermediate_size   = intermediate_size
    encoder_config.encoder_width = vision_width
    encoder_config.add_cross_attention = True
    encoder_config.cross_attention_freq = cross_attention_freq
    encoder_config.query_length = num_query_token

    qformer = _lavis.BertLMHeadModel(config=encoder_config)
    query_tokens = nn.Parameter(
        torch.zeros(1, num_query_token, encoder_config.hidden_size))
    query_tokens.data.normal_(mean=0.0, std=encoder_config.initializer_range)
    return qformer, query_tokens


def _strip_qformer_for_blip2(qformer):
    """EVCap 标准操作: 删 text 端不用的部分,省参数省显存"""
    qformer.cls = None
    qformer.bert.embeddings.word_embeddings = None
    qformer.bert.embeddings.position_embeddings = None
    for layer in qformer.bert.encoder.layer:
        layer.output = None       # text 端 FFN 删掉
        layer.intermediate = None  # text 端 FFN 删掉
        # 只剩 layer.intermediate_query / layer.output_query 走 query
    return qformer


# ============================================================
# BLIP2QFormer (image branch, 用于 build_visual_tokens)
# ============================================================
class BLIP2QFormer(nn.Module):
    """
    [REFACTOR-v6.4] 直接用 EVCap/LAVIS 官方 BertLMHeadModel,
    不再手写复现 BertLayer。

    用法:
        qformer = BLIP2QFormer(num_queries=32, num_layers=12, ...)
        qformer.load_pretrained("blip2_pretrained_flant5xxl.pth")
        out_768 = qformer(eva_patches_1408)            # [B, 32, 768] (过 output_proj)
        raw_768 = qformer.forward_raw(eva_patches)      # [B, 32, 768] (不过 output_proj)
        ext_768 = qformer.forward_with_queries(init_q_768, eva_patches)  # 用外部 init q
    """
    BLIP2_BERT_DIM = QFORMER_HIDDEN
    BLIP2_VISUAL_DIM = EVA01_DIM

    def __init__(self, num_queries=32, num_layers=12, num_heads=12,
                 intermediate_size=3072, cross_attention_freq=2):
        super().__init__()
        self.num_queries = num_queries
        self.hidden_size = self.BLIP2_BERT_DIM
        self.vis_dim = self.BLIP2_VISUAL_DIM

        # ── 用 EVCap 的工厂 ──
        self.bert_model, self.query_tokens = _build_lavis_qformer(
            num_query_token=num_queries,
            vision_width=self.vis_dim,
            hidden_size=self.hidden_size,
            num_hidden_layers=num_layers,
            num_attention_heads=num_heads,
            intermediate_size=intermediate_size,
            cross_attention_freq=cross_attention_freq,
        )
        # EVCap 原版去重: word_embeddings / position_embeddings / layer.output / intermediate / cls
        self.bert_model = _strip_qformer_for_blip2(self.bert_model)

        # ── BLIP-2 视觉 LayerNorm (就是 EVCap blip2.py 里的 self.ln_vision) ──
        self.ln_vision = nn.LayerNorm(self.vis_dim)

        # ── 输出投影 (保持原 v6.3 接口, 给 vis_proj_A 之前过一道) ──
        D = self.hidden_size
        self.norm = nn.LayerNorm(D)
        self.output_proj = nn.Sequential(nn.Linear(D, D), nn.LayerNorm(D))
        nn.init.xavier_uniform_(self.output_proj[0].weight)
        nn.init.zeros_(self.output_proj[0].bias)

        # ── 外部 init query 投影 (forward_with_queries 用) ──
        self.query_in_proj = nn.Linear(D, D, bias=True)
        nn.init.xavier_uniform_(self.query_in_proj.weight)
        nn.init.zeros_(self.query_in_proj.bias)

        ca_layers = num_layers // cross_attention_freq
        print(f"  BLIP2QFormer (LAVIS): {num_layers}层, {num_queries}q, "
              f"cross_attn={ca_layers}层(freq={cross_attention_freq})")

    # ── 别名属性, 让 model.py 现有的 .encoder.layers / .has_cross_attn 等访问继续工作 ──
    @property
    def encoder(self):
        """老接口: 让 layer.has_cross_attn / layer.cross_attn 之类的能继续访问"""
        bert_encoder = self.bert_model.bert.encoder
        if not hasattr(bert_encoder, "layers"):
            # EVCap 命名是 .layer (单数), 我们的 v6.3 老代码用 .layers (复数)
            bert_encoder.layers = bert_encoder.layer
        for L in bert_encoder.layer:
            if not hasattr(L, "has_cross_attn"):
                L.has_cross_attn = getattr(L, "has_cross_attention", False)
            # cross_attn / self.cross_attn 风格的访问 (老 unfreeze_for_stage2_lite)
            if not hasattr(L, "cross_attn") and hasattr(L, "crossattention"):
                L.cross_attn = L.crossattention
        return bert_encoder

    # ── Pretrained 权重加载 (LAVIS blip2_pretrained.pth) ──
    def load_pretrained(self, checkpoint_path):
        """
        从 LAVIS 官方 BLIP-2 ckpt 加载权重. 关键 keys:
            - query_tokens
            - Qformer.bert.*  (官方格式)
            - ln_vision.* / Qformer.ln_vision.*
        """
        if not os.path.exists(checkpoint_path):
            print(f"  ⚠️  BLIP2权重不存在: {checkpoint_path}")
            return
        patch_torch_load_safety()
        ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        state_dict = ckpt.get('model', ckpt)

        # ── 1. query_tokens ──
        n_qt = 0
        if 'query_tokens' in state_dict:
            qt = state_dict['query_tokens']
            if qt.shape == self.query_tokens.shape:
                self.query_tokens.data.copy_(qt)
                n_qt = qt.numel()
            elif qt.shape[-1] == self.hidden_size:
                # query 数不同 → 沿 query 维 1D 插值
                n_pre = qt.shape[1]
                n_mod = self.num_queries
                if n_mod != n_pre:
                    qt = F.interpolate(
                        qt.permute(0, 2, 1).float(),
                        size=n_mod, mode='linear', align_corners=True
                    ).permute(0, 2, 1)
                self.query_tokens.data.copy_(qt.to(self.query_tokens.dtype))
                n_qt = qt.numel()

        # ── 2. ln_vision ──
        n_ln = 0
        for key in ('Qformer.ln_vision', 'ln_vision'):
            if f'{key}.weight' in state_dict:
                w = state_dict[f'{key}.weight']
                if w.shape == self.ln_vision.weight.shape:
                    self.ln_vision.weight.data.copy_(w)
                    n_ln += w.numel()
                    if f'{key}.bias' in state_dict:
                        b = state_dict[f'{key}.bias']
                        if b.shape == self.ln_vision.bias.shape:
                            self.ln_vision.bias.data.copy_(b)
                            n_ln += b.numel()
                break

        # ── 3. Qformer 层权重: 直接 load_state_dict(strict=False) ──
        # EVCap ckpt 内部 key 是 "Qformer.bert.encoder.layer.X.attention..."
        # 我们的 self.bert_model 对应 "bert.encoder.layer.X.attention..."
        # 所以剥掉 "Qformer." 前缀
        bert_sd = {}
        for k, v in state_dict.items():
            if k.startswith('Qformer.'):
                new_k = k[len('Qformer.'):]
                bert_sd[new_k] = v
        if bert_sd:
            try:
                msg = self.bert_model.load_state_dict(bert_sd, strict=False)
                missing = len(getattr(msg, 'missing_keys', []))
                unexpected = len(getattr(msg, 'unexpected_keys', []))
                # 估算实际加载参数量 (按 key 在 bert_sd 里的算)
                own = dict(self.bert_model.named_parameters())
                n_loaded = sum(
                    own[k].numel() for k in bert_sd
                    if k in own and own[k].shape == bert_sd[k].shape
                )
                print(f"  ✅ Q-Former (LAVIS): query_tokens={n_qt} ln_vision={n_ln} "
                      f"bert_params≈{n_loaded/1e6:.2f}M  "
                      f"missing={missing} unexpected={unexpected}")
            except Exception as e:
                print(f"  ⚠️  Q-Former state_dict 加载部分失败: {e}")

    # ── 冻结策略 (与 v6.3 老接口对齐) ──
    def freeze_query_and_kv_only(self):
        """只解冻: query_tokens, 所有 cross_attention 的 K/V, output_proj, query_in_proj"""
        for p in self.parameters():
            p.requires_grad = False
        self.query_tokens.requires_grad = True
        for layer in self.bert_model.bert.encoder.layer:
            if getattr(layer, "has_cross_attention", False):
                # EVCap 的命名: layer.crossattention.self.key / value
                for p in layer.crossattention.self.key.parameters():
                    p.requires_grad = True
                for p in layer.crossattention.self.value.parameters():
                    p.requires_grad = True
        for p in self.output_proj.parameters():
            p.requires_grad = True
        for p in self.query_in_proj.parameters():
            p.requires_grad = True

    freeze_bert = freeze_query_and_kv_only

    # ── 核心 forward ──
    def _run_bert(self, init_query, vis_1408, vis_mask=None):
        """
        统一 forward 内核. 走 EVCap 标准路径:
          self.Qformer.bert(
              query_embeds=init_query,
              encoder_hidden_states=ln_vision(vis),
              encoder_attention_mask=vis_mask,
              return_dict=True)
        """
        B = vis_1408.shape[0]
        device = vis_1408.device

        # 数值清洗 (跟原 v6.3 一致)
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
        # last_hidden_state shape: [B, num_queries, hidden_size]
        return outputs.last_hidden_state

    def forward(self, visual_features_1408, visual_mask=None):
        """标准入口: 用模块自带的 query_tokens, 输出过 output_proj"""
        B = visual_features_1408.shape[0]
        queries = self.query_tokens.expand(B, -1, -1)
        out = self._run_bert(queries, visual_features_1408, visual_mask)
        out = self.norm(out)
        return self.output_proj(out)

    def forward_raw(self, visual_features_1408, visual_mask=None):
        """不过 output_proj 的版本"""
        B = visual_features_1408.shape[0]
        queries = self.query_tokens.expand(B, -1, -1)
        return self._run_bert(queries, visual_features_1408, visual_mask)

    def forward_with_queries(self, init_queries_768, visual_features_1408, visual_mask=None):
        """
        [V6.3-NAME-ATTR 关键] 用外部传入的 init_queries (已是 768 维) 作为 query_embeds.
        例如把 PKL 检索到的 name embedding 当 init,跑一次 cross-attention 精炼.
        """
        # 先过 query_in_proj 让外部嵌入进入 Qformer 的 query 子空间
        queries = self.query_in_proj(init_queries_768.float())
        out = self._run_bert(queries, visual_features_1408, visual_mask)
        out = self.norm(out)
        return self.output_proj(out)
