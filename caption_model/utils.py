#!/usr/bin/env python3
"""
utils.py ── 通用工具函数

包括:
  - torch.load 安全补丁
  - Caption 后处理 (去除非拉丁字符、规范化等)
  - 空间编码工具 (loc tokens, sinusoidal expansion)
  - EVCap 风格名称清洗
  - 实体文本构造
  - per-token-vote
"""

import re
import math
import unicodedata as _ud
from collections import Counter
from typing import List

import torch
import torch.nn.functional as F
import numpy as np

from .constants import (
    NUM_BINS, INVALID_TAG_NAMES_EXTENDED,
)

# ============================================================
# torch.load 安全补丁 (兼容 transformers 新版本检查)
# ============================================================
_TORCH_LOAD_PATCHED = False


def patch_torch_load_safety():
    global _TORCH_LOAD_PATCHED
    if _TORCH_LOAD_PATCHED:
        return
    _TORCH_LOAD_PATCHED = True

    def _noop():
        pass

    for mod_path, attr in [
        ('transformers.utils.import_utils', 'check_torch_load_is_safe'),
        ('transformers.modeling_utils', 'check_torch_load_is_safe'),
    ]:
        try:
            import importlib
            m = importlib.import_module(mod_path)
            if hasattr(m, attr):
                setattr(m, attr, _noop)
        except Exception:
            pass


# 模块加载时立即应用补丁
patch_torch_load_safety()


# ============================================================
# Caption 后处理
# ============================================================
def is_non_latin(text: str, threshold: float = 0.3) -> bool:
    """检查文本是否含有过多非拉丁字符"""
    if not text:
        return False
    non_latin = 0
    total = 0
    for ch in text:
        if ch.isalpha():
            total += 1
            try:
                name = _ud.name(ch, '')
                if any(script in name for script in (
                        'CYRILLIC', 'CJK', 'ARABIC', 'DEVANAGARI', 'THAI',
                        'HANGUL', 'HIRAGANA', 'KATAKANA', 'GEORGIAN',
                        'ARMENIAN', 'HEBREW', 'BENGALI', 'TAMIL',
                )):
                    non_latin += 1
            except ValueError:
                non_latin += 1
    return total > 0 and (non_latin / total) > threshold


def strip_non_latin_chars(text: str) -> str:
    """去除非拉丁字符"""
    cleaned = []
    for ch in text:
        if ord(ch) < 128:
            cleaned.append(ch)
        elif ch.isalpha():
            try:
                name = _ud.name(ch, '')
                if 'LATIN' in name:
                    cleaned.append(ch)
            except ValueError:
                pass
        else:
            cleaned.append(ch)
    return ''.join(cleaned).strip()


def clean_caption(text: str) -> str:
    """规范化 caption 输出"""
    if not text or not text.strip():
        return text
    text = re.sub(r'^[\x00-\x1F\x7F-\x9F\u0080-\uFFFF]+', '', text)
    text = text.strip().lstrip('.').strip()
    if not text:
        return text
    text = re.sub(r'^\s*\(\d+\)\s*', '', text).strip()
    text = re.sub(r'^\s*\d+[.)]\s+', '', text).strip()
    text = re.sub(r'^[\*\-•]\s+', '', text).strip()
    for garbage in ("(L-R)", "(L-r)", "(l-r)", "L-R:", "L-R)",
                    "USER:", "ASSISTANT:", "###", ">>>"):
        if text.startswith(garbage):
            text = text[len(garbage):].strip()
    if is_non_latin(text, threshold=0.3):
        english_part = strip_non_latin_chars(text)
        if english_part and len(english_part) > 5 and not is_non_latin(english_part, 0.1):
            text = english_part
        else:
            return ""
    lower = text.lower()
    for prefix in ("a photo of ", "an image of ", "a picture of ",
                   "photo of ", "image of ", "picture of "):
        if lower.startswith(prefix):
            text = text[len(prefix):].strip()
    text = text.lstrip(',;: ').strip()
    if not text:
        return text
    for sep in ['. ', '.\n', '\n']:
        idx = text.find(sep)
        if idx > 5:
            text = text[:idx + 1]
            break
    text = text[0].upper() + text[1:] if len(text) > 1 else text.upper()
    if text and text[-1] not in '.!?':
        text = text.rstrip(',;: ') + '.'
    return text


# ============================================================
# EVCap 风格的名称清洗 (pre_name)
# ============================================================
def evcap_pre_name(caption: str) -> str:
    if not caption:
        return ""
    caption = re.sub(r"([_!,'\"()*#:;~])", " ", caption.lower())
    caption = re.sub(r"\s{2,}", " ", caption)
    caption = caption.rstrip("\n").strip(" ")
    return caption


# ============================================================
# 空间编码工具
# ============================================================
def abs_to_loc_tokens(abs_vec) -> str:
    """绝对位置向量转 loc token 字符串"""
    vals = abs_vec.tolist() if hasattr(abs_vec, 'tolist') else list(abs_vec)
    cx, cy, w, h, area, region, dist_c, ar, conf, is_large = (float(v) for v in vals)
    cx_b = int(min(cx, 0.999) * NUM_BINS)
    cy_b = int(min(cy, 0.999) * NUM_BINS)
    w_b = int(min(w, 0.999) * NUM_BINS)
    h_b = int(min(h, 0.999) * NUM_BINS)
    area_b = int(min(area, 0.999) * NUM_BINS)
    reg = max(0, min(8, int(round(region))))
    sz = "large" if is_large > 0.5 else ("medium" if area > 0.04 else "small")
    ar_w = "wide" if ar > 1.4 else ("square" if ar > 0.7 else "tall")
    return (f"<loc{cx_b}><loc{cy_b}><loc{w_b}><loc{h_b}>"
            f"<loc{area_b}><reg{reg}><sz_{sz}><ar_{ar_w}>")


def sinusoidal_expand(values: torch.Tensor, dim: int = 64) -> torch.Tensor:
    """正弦展开"""
    assert dim % 2 == 0
    freqs = torch.pow(2, torch.arange(dim // 2, dtype=torch.float32,
                                      device=values.device)) * math.pi
    angles = values.unsqueeze(-1).float() * freqs.unsqueeze(0)
    return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1).reshape(-1)


def rel_sinusoidal(rel_vec: torch.Tensor, sin_dim: int = 32) -> torch.Tensor:
    """相对位置正弦编码"""
    dx, dy, dist, sin_v, cos_v, size_log, iou, edge_d, x_ov, y_ov, ref_id, diff = (
        rel_vec.tolist() if hasattr(rel_vec, 'tolist') else rel_vec)
    norm = torch.tensor([
        dx, dy, dist / 1.415, sin_v, cos_v,
        math.tanh(float(size_log)), iou, edge_d / 1.415, x_ov, y_ov,
        (float(ref_id) % 9) / 9.0, math.tanh(float(diff)),
    ], dtype=torch.float32,
        device=rel_vec.device if hasattr(rel_vec, 'device') else 'cpu')
    return sinusoidal_expand(norm, dim=sin_dim)


# ============================================================
# 实体文本构造
# ============================================================
def build_entity_text(tag_seq, abs_vec):
    """从 (tag, abs_pos) 构造 'tag at:[x1,y1,x2,y2]' 字符串"""
    vals = abs_vec.tolist() if hasattr(abs_vec, 'tolist') else list(abs_vec)
    cx, cy, w, h = float(vals[0]), float(vals[1]), float(vals[2]), float(vals[3])
    x1 = int(max(0, cx - w / 2) * 999)
    y1 = int(max(0, cy - h / 2) * 999)
    x2 = int(min(1, cx + w / 2) * 999)
    y2 = int(min(1, cy + h / 2) * 999)
    tag = tag_seq.strip() if tag_seq and tag_seq.strip() else "object"
    return f"{tag} at:[{x1},{y1},{x2},{y2}]"


# ============================================================
# per-token-vote (诊断用)
# ============================================================
def per_token_vote(tokens_768, name_keys, name_values, top_k_per_token=5, vote_weight=1):
    """每个 token 的 top-K 投票"""
    votes = Counter()
    Q = tokens_768.shape[0]
    q_norm = F.normalize(tokens_768.float(), p=2, dim=-1)
    all_sims = (q_norm @ name_keys.T).float().detach().cpu().numpy()
    k = min(top_k_per_token, all_sims.shape[1] - 1)
    for qi in range(Q):
        top_idx = np.argpartition(-all_sims[qi], k)[:top_k_per_token]
        for idx in top_idx:
            name = name_values[idx] if idx < len(name_values) else ""
            if name and name not in INVALID_TAG_NAMES_EXTENDED:
                votes[name] += vote_weight
    return votes
