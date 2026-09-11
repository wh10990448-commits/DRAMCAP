#!/usr/bin/env python3
"""
soft_retriever.py ── LightSoftRetriever (轻量软检索器)

通过多流 cross-attention 把 query tokens 与 memory bank 对齐,
返回融合后的 tokens + 各流的 attention 权重。
"""

from typing import Optional, Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .constants import INVALID_TAG_NAMES_EXTENDED


class LightSoftRetriever(nn.Module):
    """
    轻量级软检索器

    每个 stream (name/color/size/...) 在共享的 cross-attention 上
    叠加一个可学习的 stream_bias, 然后用可学习的 stream_weights
    做加权融合。
    """

    _DEFAULT_STREAMS: List[str] = ['name', 'color', 'size', 'material', 'shape']

    def __init__(self, dim: int = 768, num_heads: int = 8,
                 streams: Optional[List[str]] = None):
        super().__init__()
        self.dim = dim
        self.streams = list(streams) if streams is not None else list(self._DEFAULT_STREAMS)

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads,
            batch_first=True, dropout=0.0,
        )
        self.stream_bias = nn.Parameter(
            torch.zeros(len(self.streams), 1, 1, dim))
        nn.init.normal_(self.stream_bias, std=0.02)
        self.stream_weights = nn.Parameter(
            torch.ones(len(self.streams)))
        self.norm = nn.LayerNorm(dim)

        self._bank: Optional[Dict] = None
        self.last_name_attn: Optional[torch.Tensor] = None
        self.current_gate_value: float = 0.0

        print(f"  LightSoftRetriever: {len(self.streams)} 流  streams={self.streams}")

    # ------------------------------------------------------------
    # Memory Bank 管理
    # ------------------------------------------------------------
    def set_memory_bank(self, bank: Dict):
        """
        设置 memory bank, 并按 bank.keys() 重组 streams 顺序
        会保留原有 stream 的 bias / weight (按 stream 名匹配)
        """
        self._bank = bank
        new_streams = [k for k in bank.keys()]
        if new_streams == self.streams:
            return

        old_n = len(self.streams)
        new_n = len(new_streams)
        device = self.stream_bias.device if self.stream_bias is not None else torch.device('cpu')
        dtype = self.stream_bias.dtype if self.stream_bias is not None else torch.float32

        old_bias_data = self.stream_bias.data if self.stream_bias is not None else None
        old_weight_data = self.stream_weights.data if self.stream_weights is not None else None
        old_stream_index = {s: i for i, s in enumerate(self.streams)}

        new_bias = torch.zeros(new_n, 1, 1, self.dim, device=device, dtype=dtype)
        new_weights = torch.ones(new_n, device=device, dtype=dtype)
        nn.init.normal_(new_bias, std=0.02)

        for j, sname in enumerate(new_streams):
            if old_bias_data is not None and sname in old_stream_index:
                i = old_stream_index[sname]
                if i < old_n:
                    new_bias[j] = old_bias_data[i]
                    new_weights[j] = old_weight_data[i]

        self.stream_bias = nn.Parameter(new_bias)
        self.stream_weights = nn.Parameter(new_weights)
        self.streams = new_streams
        print(f"  [LightSoftRetriever] set_memory_bank: "
              f"{old_n} 流 → {new_n} 流  {new_streams}")

    def update_stream_val_embeds(self, stream_name: str, embeds: torch.Tensor):
        """更新某个流的 val_embeds (例如 LLM 学到了新的属性向量)"""
        if self._bank is not None and stream_name in self._bank:
            cur = self._bank[stream_name]['val_embeds']
            n = min(len(embeds), cur.shape[0])
            self._bank[stream_name]['val_embeds'][:n] = embeds[:n].to(cur.dtype)

    # ------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------
    def forward(self, query_tokens, memory_bank=None):
        bank = memory_bank if memory_bank is not None else self._bank
        assert bank is not None, "请先调用 set_memory_bank()"

        B, L, D = query_tokens.shape
        device = query_tokens.device
        outputs = []
        dbg = {}

        for i, sname in enumerate(self.streams):
            if sname not in bank:
                outputs.append(torch.zeros(B, L, D, device=device,
                                           dtype=query_tokens.dtype))
                continue
            mk = bank[sname]['keys'].to(device).to(query_tokens.dtype)
            mv = bank[sname]['val_embeds'].to(device).to(query_tokens.dtype)
            mk = mk.unsqueeze(0).expand(B, -1, -1)
            mv = mv.unsqueeze(0).expand(B, -1, -1)
            bias = self.stream_bias[i].expand(B, L, -1).to(query_tokens.dtype)
            q = query_tokens + bias
            out, attn_w = self.cross_attn(q, mk, mv)
            outputs.append(out)
            dbg[sname] = attn_w

        w = torch.softmax(self.stream_weights, dim=0)
        fused = sum(w[i] * outputs[i] for i in range(len(self.streams)))
        out = self.norm(fused + query_tokens)

        self.last_name_attn = dbg.get('name')
        self.current_gate_value = float(w[0].item())
        return out, dbg

    # ------------------------------------------------------------
    # 解析 attention 得到 top-K 名称 (诊断用)
    # ------------------------------------------------------------
    def get_top_names(self, dbg, bank=None, topk=8):
        b = bank if bank is not None else self._bank
        if 'name' not in dbg or b is None or 'name' not in b:
            return []
        attn_w = dbg['name']
        texts = b['name']['texts']
        scores = attn_w.mean(dim=1)
        result = []
        for s in scores:
            seen = set()
            top_idx = s.topk(min(topk * 3, len(texts))).indices.tolist()
            names_b = []
            for i in top_idx:
                name = texts[i]
                if name not in INVALID_TAG_NAMES_EXTENDED and name not in seen:
                    names_b.append(name)
                    seen.add(name)
                    if len(names_b) >= topk:
                        break
            result.append(names_b)
        return result
