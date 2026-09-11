#!/usr/bin/env python3
"""
build_attr_memory.py  v6.9.2  —  统一属性记忆库构建器 (适配 v3 classify)
═════════════════════════════════════════════════════════════════════
v6.9.2 关键改动 (相对 v6.9.1):
  ── 适配 classify_vg_attrs_v3 输出 ──
  v3 classify 加了 caption 词频/启发式过滤后,
  输出 ~94 个 dim 候选 (含 temperature/taste/lighting 等 caption 高频 dim)
  v6.9.1 白名单只放 43 个, 把 v3 精心分类的 70+ 个 dim 又当垃圾丢掉, 不合理.

  v6.9.2 修复:
  1) ★ 重写 CAPTION_FRIENDLY_DIMS, 跟 v3 输出严格对齐 (61 个高质量 dim)
     - 命名风格统一: VAW 原生用空格 ('hair color' / 'face expression')
     - 修 v6.9.1 的 3 个 bug: 'face_expression'/'hair_color'/'hair_length'
       (下划线版本 VAW 字符串不等于, 直接被丢)
  2) ★ 收纳 v3 新增 caption-friendly dim:
     - 物理感知: temperature, taste, wetness
     - 食物专属: doneness, cooking_state, freshness, ripeness, fullness
     - 环境:    lighting, time_of_day, season, weather
     - 人物:    emotion, hairstyle
     - 车辆:    motion_state, door_state, light_state
     - 状态:    openness, lit_state, completion
     - 视觉:    finish, reflectivity
  3) ★ 主动砍掉 34 个低 caption 价值的 dim:
     accessibility, alignment, angle, body_position, curvature, density,
     distribution, edge_type, eye_state, face pose, facial_feature,
     formality, functionality, hand_gesture, letter color, luminosity,
     mouth_state, operational, profile, quality, race, rarity,
     repetition, saturation, size comparison, skin color, smell, sound,
     speed, stack, symmetry, transparency_level, usage, value_judgment

  v6.9.1 继承不变 (双层过滤):
  ┌─────────────────────────┬──────────────┬────────────┐
  │ 过滤项                   │  PACO + VAW  │  VG(严格)   │
  ├─────────────────────────┼──────────────┼────────────┤
  │ caption 词频门槛         │   >= 30      │  >= 50     │
  │ dim 白名单               │  ~61 dim     │  必在 VAW dim 集合内 │
  │ 每 (obj,dim,av) box 上限 │   80         │  30        │
  │ 每 (obj,dim,av) box 下限 │   1          │  3 (后置过滤) │
  └─────────────────────────┴──────────────┴────────────┘
"""

import os, sys, gc, re, json, pickle, argparse, urllib.request

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _THIS_DIR)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from typing import List, Dict, Tuple, Optional, Set
from collections import defaultdict, Counter

# ════════════════════════════════════════════════════════════════════
# 路径配置
# ════════════════════════════════════════════════════════════════════
NAME_PKL_PATH = "/root/autodl-tmp/wh5319/silip_clip_caption/image_captioning_final/ext_memory_lvis.pkl"
EVA_CLIP_PATH = "/root/autodl-tmp/wh5319/silip_clip_caption/image_captioning_final/eva_vit_g.pth"
EVA_VIT_G_URL = "https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/eva_vit_g.pth"
BLIP2_CKPT = "/root/autodl-tmp/wh5319/silip_clip_caption/model/blip2_pretrained_flant5xxl.pth"
VAW_DIR = "/root/autodl-tmp/wh5319/all_data/data"
VG_IMAGES_DIR = "/root/autodl-tmp/wh5319/all_data/VG"
VG_ATTRIBUTES_JSON = "/root/autodl-tmp/wh5319/all_data/VG/attributes.json"
VG_IMAGE_DATA_JSON = "/root/autodl-tmp/wh5319/all_data/VG/image_data.json"
VG_ATTR_TYPES_JSON = "/root/autodl-tmp/wh5319/all_data/VG/vg_attribute_types.json"

PACO_JSON = "/root/autodl-tmp/wh5319/silip_clip_caption/image_captioning_final/paco_data/paco_lvis_v1_train.json"
COCO_IMAGES = ["/root/autodl-tmp/wh5319/all_data/coco_2014/train2014",
               "/root/autodl-tmp/wh5319/all_data/coco_2014/val2014",
               "/root/autodl-tmp/wh5319/all_data/coco2017"]
OUTPUT_DIR = "/root/autodl-tmp/wh5319/silip_clip_caption/image_captioning_final/attr_memory"

COCO_CAPTION_JSON = "/root/autodl-tmp/wh5319/all_data/coco_2014/annotations/captions_train2014.json"

# ════════════════════════════════════════════════════════════════════
# 超参
# ════════════════════════════════════════════════════════════════════
DEVICE = "cuda"
NUM_QUERY = 32
BATCH_SIZE = 8
BBOX_MARGIN = 0.1
MIN_IMAGES = 1

TOPK_ATTRS = 100
MIN_RATIO = 0.01
MIN_COUNT = 1
MAX_PER_DIM = 16
MAX_PER_ATTR = 80

MIN_DIMS_PER_OBJ = 2
CAPTION_FREQ_THRESHOLD = 30

# v6.9.1 VG 专用过滤
VG_CAPTION_FREQ_THRESHOLD = 50
VG_MAX_BOX_PER_ATTR = 30
VG_MIN_BOX_PER_ATTR = 3
VG_RESTRICT_DIMS_TO_VAW = True

# ════════════════════════════════════════════════════════════════════
# v6.9.2 关键: Caption-friendly dim 白名单 (跟 v3 classify 输出对齐)
# ════════════════════════════════════════════════════════════════════
# 对齐验证 (跑 check_dim_match 脚本得出):
#   v3 classify 输出 dim: 94 个
#   v6.9.2 白名单:        61 个
#   v6.9.2 ∩ v3 输出:     60 个 ← 实际入库的核心
#   v6.9.2 但 v3 不出:    1 个 ('opacity', PACO 映射产出)
# ════════════════════════════════════════════════════════════════════
CAPTION_FRIENDLY_DIMS: Set[str] = {
    # ── PACO 核心 5 dim (PACO_TYPE_TO_DIM 映射后产出) ────────
    'color', 'material', 'pattern', 'shape', 'opacity',

    # ── VAW 原 42 中 caption 高频 (★ 注意空格风格, 跟 VAW 一致) ─
    'size', 'height', 'length', 'width', 'thickness', 'weight', 'depth',
    'fatness',
    'state', 'cleanliness', 'maturity', 'newness',
    'texture', 'hardness', 'orientation', 'tone',
    'pose', 'activity', 'gender',
    'face expression',                   # ★ v6.9.1 写错成 'face_expression'
    'hair color', 'hair length',         # ★ v6.9.1 写错成下划线版
    'wearing color', 'wearing accessories', 'sport activity',
    'hand movement', 'closeness', 'place', 'location',
    'weather', 'machine state', 'opaqeness',
    'brightness', 'color quantity',

    # ── v3 新增 caption-friendly dim (VG 独有) ───────────────
    # 物理感知
    'temperature',          # hot/cold/warm
    'wetness',              # wet/dry/damp
    'taste',                # sweet/sour/spicy

    # 状态细分
    'fullness',             # full/empty
    'freshness',            # fresh/stale/rotten
    'ripeness',             # ripe/unripe
    'openness',             # open/closed
    'lit_state',            # lit/glowing/burning
    'completion',           # complete/whole

    # 视觉细分
    'finish',               # glossy/matte
    'reflectivity',         # reflective/shiny

    # 食物专属
    'doneness',             # raw/cooked/burnt
    'cooking_state',        # fried/baked/grilled
    'preparation',          # sliced/chopped

    # 人物专属
    'emotion',              # happy/sad/angry
    'hairstyle',            # curly/straight/braided

    # 车辆/物品
    'motion_state',         # moving/stationary/parked
    'door_state',           # open/closed
    'light_state',          # on/off/headlights

    # 环境
    'lighting',             # well-lit/dim/natural
    'time_of_day',          # morning/evening/night
    'season',               # spring/summer/winter

    # ── 主动砍掉的 v3 输出 (低 caption 价值, 共 34 个) ────────
    # accessibility, alignment, angle, body_position, curvature,
    # density, distribution, edge_type, eye_state, face pose,
    # facial_feature, formality, functionality, hand_gesture,
    # letter color, luminosity, mouth_state, operational, profile,
    # quality, race, rarity, repetition, saturation, size comparison,
    # skin color, smell, sound, speed, stack, symmetry,
    # transparency_level, usage, value_judgment
}

# ════════════════════════════════════════════════════════════════════
# 黑名单 (跟 v6.8 一致)
# ════════════════════════════════════════════════════════════════════
ATTR_BLACKLIST = {
    "nice", "beautiful", "ugly", "good", "bad", "cool", "cute", "pretty", "fancy",
    "normal", "regular", "typical", "common", "other", "different", "same",
    "similar", "interesting", "boring", "unusual",
    "other(color)", "other(pattern_marking)", "other(material)",
    "other(transparency)", "other(shape)",
    "unknown", "cannot_tell_from_the_image",
}

PACO_TYPE_TO_DIM = {
    "color": "color",
    "material": "material",
    "pattern_marking": "pattern",
    "transparency": "opacity",
    "shape": "shape",
    "reflectance": "opacity",
    "patterns_markings": "pattern",
    "patterns": "pattern",
}

SKIP_SUBTYPES: Set[str] = set()


# ════════════════════════════════════════════════════════════════════
# COCO caption 词频统计
# ════════════════════════════════════════════════════════════════════
def build_caption_word_freq(coco_caption_json: str) -> Optional[Counter]:
    if not os.path.exists(coco_caption_json):
        print(f"  ⚠️ COCO caption json 不存在: {coco_caption_json}")
        print(f"     caption 词频过滤将被禁用")
        return None
    print(f"  [caption-freq] 统计 COCO caption 词频 ...")
    print(f"    源文件: {coco_caption_json}")
    with open(coco_caption_json) as f:
        data = json.load(f)
    counter = Counter()
    n_anns = 0
    for ann in data.get('annotations', []):
        n_anns += 1
        cap = ann.get('caption', '').lower().strip().rstrip('.').rstrip(',').rstrip('!')
        words = re.split(r"[\s,;.!?\"'\(\)]+", cap)
        words = [w for w in words if w]
        counter.update(words)
    print(f"    → {n_anns:,} 条 caption, {len(counter):,} 个唯一词, 总词数 {sum(counter.values()):,}")
    print(f"    前 20 高频词: {[w for w, _ in counter.most_common(20)]}")
    return counter


# ════════════════════════════════════════════════════════════════════
# 加载并合并 VAW + VG attribute_types.json
# ════════════════════════════════════════════════════════════════════
def load_vaw_attr_types(vaw_dir: str) -> Dict[str, List[str]]:
    fp = os.path.join(vaw_dir, "attribute_types.json")
    if not os.path.exists(fp):
        print(f"  ⚠️ 找不到 VAW {fp}")
        return {}
    with open(fp) as f:
        d = json.load(f)
    print(f"  [VAW types] 加载 {len(d)} 个 dim, {sum(len(v) for v in d.values())} 个 attr 词")
    print(f"    VAW dim 命名 (前 10): {sorted(d.keys())[:10]}")
    return d


def load_vg_attr_types(path: str) -> Dict[str, List[str]]:
    if not os.path.exists(path):
        print(f"  ⚠️ 找不到 VG attribute_types: {path}")
        print(f"     提示: 跑 classify_vg_attrs_v3.py 先生成此文件")
        return {}
    with open(path) as f:
        d = json.load(f)
    print(f"  [VG types] 加载 {len(d)} 个 dim, {sum(len(v) for v in d.values())} 个 attr 词")
    print(f"    VG dim 命名 (前 10): {sorted(d.keys())[:10]}")
    return d


def merge_attr_types(vaw_types: Dict[str, List[str]],
                     vg_types: Dict[str, List[str]],
                     vaw_priority: bool = True) -> Tuple[Dict[str, str], Dict[str, List[str]]]:
    unified_attr2dim: Dict[str, str] = {}
    unified_dim2attrs: Dict[str, List[str]] = defaultdict(list)
    seen_attr: Set[str] = set()

    for vaw_dim, attrs in vaw_types.items():
        d = vaw_dim.lower().strip()
        for a in attrs:
            av = a.lower().strip()
            if not av or av in ATTR_BLACKLIST:
                continue
            if av in seen_attr:
                continue
            seen_attr.add(av)
            unified_attr2dim[av] = d
            unified_dim2attrs[d].append(av)

    n_vaw = len(seen_attr)
    n_dim_from_vaw = len(unified_dim2attrs)

    n_vg_added_attr = 0
    n_vg_dim_new = 0
    for vg_dim, attrs in vg_types.items():
        d = vg_dim.lower().strip()
        was_new_dim = d not in unified_dim2attrs
        for a in attrs:
            av = a.lower().strip()
            if not av or av in ATTR_BLACKLIST:
                continue
            if vaw_priority and av in seen_attr:
                continue
            seen_attr.add(av)
            unified_attr2dim[av] = d
            unified_dim2attrs[d].append(av)
            n_vg_added_attr += 1
        if was_new_dim and d in unified_dim2attrs:
            n_vg_dim_new += 1

    print(f"\n  [合并结果]")
    print(f"    VAW 贡献: {n_vaw:,} attr 词, {n_dim_from_vaw} 个 dim")
    if vg_types:
        print(f"    VG  补充: {n_vg_added_attr:,} attr 词 (新 dim {n_vg_dim_new} 个)")
    print(f"    合并总计: {len(seen_attr):,} attr 词, {len(unified_dim2attrs)} 个 dim")
    print(f"\n  [前 20 个 dim 内 attr 数]")
    sorted_dims = sorted(unified_dim2attrs.items(), key=lambda x: -len(x[1]))
    for dn, attrs in sorted_dims[:20]:
        print(f"    {dn:25s}: {len(attrs):>4} attr 词")
    if len(sorted_dims) > 20:
        print(f"    ... 还有 {len(sorted_dims) - 20} 个 dim")

    return unified_attr2dim, dict(unified_dim2attrs)


# ════════════════════════════════════════════════════════════════════
# v6.9.2: dim 白名单 + caption 词频双重过滤 (加诊断对齐打印)
# ════════════════════════════════════════════════════════════════════
def filter_unified_by_whitelist_and_freq(
        unified_attr2dim: Dict[str, str],
        unified_dim2attrs: Dict[str, List[str]],
        dim_whitelist: Optional[Set[str]] = None,
        caption_freq: Optional[Counter] = None,
        freq_threshold: int = 0,
) -> Tuple[Dict[str, str], Dict[str, List[str]]]:
    print(f"\n  [v6.9.2 双重过滤]")
    n_before = len(unified_attr2dim)

    # ★ v6.9.2 新增: 诊断打印 — 源 dim 集合 vs 白名单 命名对齐情况
    src_dims = set(unified_dim2attrs.keys())
    if dim_whitelist is not None:
        in_both = src_dims & dim_whitelist
        in_src_only = src_dims - dim_whitelist
        in_wl_only = dim_whitelist - src_dims
        print(f"    [对齐诊断]")
        print(f"      源 dim 数 (VAW+VG):     {len(src_dims)}")
        print(f"      白名单 dim 数:           {len(dim_whitelist)}")
        print(f"      ∩ (实际命中):            {len(in_both)}  ← 这些 dim 的 attr 会被保留")
        print(f"      源有但白名单没收的 dim:  {len(in_src_only)}  ← 这些 attr 会被过滤掉")
        print(f"      白名单有但源没产的 dim:  {len(in_wl_only)}  ← 这些是空头 (仅 PACO 后续会产)")
        if in_wl_only and len(in_wl_only) <= 10:
            print(f"        空头具体: {sorted(in_wl_only)}")

    # 1) dim 白名单过滤
    if dim_whitelist is not None:
        filtered_attr2dim = {
            av: d for av, d in unified_attr2dim.items() if d in dim_whitelist
        }
        n_after_wl = len(filtered_attr2dim)
        n_dim_dropped = sum(1 for d in unified_dim2attrs.keys() if d not in dim_whitelist)
        dropped_dims = [d for d in unified_dim2attrs.keys() if d not in dim_whitelist]
        print(f"    [Step 1: dim 白名单]")
        print(f"      白名单大小: {len(dim_whitelist)} 个 dim")
        print(f"      源 dim 数:   {len(unified_dim2attrs)}")
        print(f"      丢弃 dim:    {n_dim_dropped} 个")
        if dropped_dims and len(dropped_dims) <= 30:
            print(f"        (具体: {sorted(dropped_dims)})")
        elif dropped_dims:
            print(f"        (前 20: {sorted(dropped_dims)[:20]} ... +{len(dropped_dims) - 20})")
        print(f"      attr 词:     {n_before:,} → {n_after_wl:,} (-{n_before - n_after_wl:,})")
    else:
        filtered_attr2dim = dict(unified_attr2dim)
        print(f"    [Step 1: dim 白名单] 已禁用")

    # 2) caption 词频过滤
    if caption_freq is not None and freq_threshold > 0:
        before2 = len(filtered_attr2dim)
        filtered_attr2dim = {
            av: d for av, d in filtered_attr2dim.items()
            if all(caption_freq.get(w, 0) >= freq_threshold for w in av.split())
        }
        n_after_freq = len(filtered_attr2dim)
        dropped_examples = [av for av in unified_attr2dim
                            if av not in filtered_attr2dim
                            and (dim_whitelist is None or unified_attr2dim[av] in dim_whitelist)][:8]
        print(f"    [Step 2: caption 词频 (>= {freq_threshold})]")
        print(f"      attr 词: {before2:,} → {n_after_freq:,} (-{before2 - n_after_freq:,})")
        if dropped_examples:
            print(f"      丢弃示例: {dropped_examples}")
    else:
        print(f"    [Step 2: caption 词频] 已禁用 (threshold={freq_threshold})")

    # 3) 重建 dim2attrs
    final_dim2attrs: Dict[str, List[str]] = defaultdict(list)
    for av, d in filtered_attr2dim.items():
        final_dim2attrs[d].append(av)

    print(f"\n  [v6.9.2 过滤后最终]")
    print(f"    总 attr 词: {len(filtered_attr2dim):,}")
    print(f"    总 dim:     {len(final_dim2attrs)}")
    sorted_dims = sorted(final_dim2attrs.items(), key=lambda x: -len(x[1]))
    for dn, attrs in sorted_dims:
        print(f"      {dn:25s}: {len(attrs):>4} attr 词")

    return filtered_attr2dim, dict(final_dim2attrs)


# ════════════════════════════════════════════════════════════════════
# v6.9.1 关键: VG 专用更严格的 attr→dim 表
# ════════════════════════════════════════════════════════════════════
def build_vg_strict_attr2dim(
        unified_attr2dim: Dict[str, str],
        caption_freq: Optional[Counter],
        vg_freq_threshold: int,
        vaw_dims: Set[str],
        restrict_to_vaw_dims: bool,
) -> Dict[str, str]:
    print(f"\n  [v6.9.1 VG 严格表构建]")
    print(f"    源 (PACO/VAW 用的): {len(unified_attr2dim):,} attr 词")
    print(f"    VG 专用词频门槛:    {vg_freq_threshold} (PACO/VAW 用的是 30)")
    print(f"    VG dim 限制在 VAW:  {restrict_to_vaw_dims}")
    if restrict_to_vaw_dims:
        print(f"    VAW dim 集合 ({len(vaw_dims)}): {sorted(vaw_dims)}")

    vg_attr2dim = {}
    n_filtered_dim = 0
    n_filtered_freq = 0

    for av, d in unified_attr2dim.items():
        if restrict_to_vaw_dims and d not in vaw_dims:
            n_filtered_dim += 1
            continue
        if caption_freq is not None and vg_freq_threshold > 0:
            if not all(caption_freq.get(w, 0) >= vg_freq_threshold for w in av.split()):
                n_filtered_freq += 1
                continue
        vg_attr2dim[av] = d

    print(f"    过滤丢 dim 不在 VAW: {n_filtered_dim:,}")
    print(f"    过滤丢 词频 < {vg_freq_threshold}: {n_filtered_freq:,}")
    print(f"    VG 严格表大小:        {len(vg_attr2dim):,} attr 词")

    return vg_attr2dim


# ════════════════════════════════════════════════════════════════════
# 双向匹配相关 (跟 v6.9.1 完全一致)
# ════════════════════════════════════════════════════════════════════
_LEADING_MODIFIERS = {
    "white", "black", "red", "blue", "green", "yellow", "orange", "purple",
    "pink", "brown", "gray", "grey", "golden", "silver", "beige", "tan",
    "navy", "maroon", "violet", "turquoise", "cyan", "magenta", "khaki",
    "light", "dark", "bright", "pale",
    "large", "small", "big", "little", "tiny", "huge", "long", "short",
    "tall", "wide", "narrow", "round", "square", "rectangular", "thin",
    "thick", "fat", "skinny", "mini", "giant", "jumbo", "compact",
    "old", "young", "new", "baby", "adult", "elderly", "junior", "senior",
    "fresh", "ripe", "rotten", "dry", "wet", "clean", "dirty", "broken",
    "shiny", "rusty", "worn", "torn",
    "wooden", "plastic", "metal", "metallic", "glass", "leather", "ceramic",
    "stone", "iron", "steel", "rubber", "paper", "concrete", "marble",
    "fabric", "cloth", "cotton", "wool", "silk", "porcelain", "bronze",
    "copper", "brass",
    "indian", "african", "american", "asian", "european", "japanese", "chinese",
    "italian", "french", "german", "english", "spanish", "mexican", "korean",
    "thai", "russian", "australian", "british", "canadian",
    "kitchen", "office", "school", "computer", "race", "sports", "tennis",
    "baseball", "soccer", "basketball", "football", "golf", "wedding",
    "rocking", "dining", "coffee", "tea", "wine", "beer", "water", "milk",
    "juice", "fruit", "vegetable", "kids", "men", "women", "boys", "girls",
    "fancy", "plain", "simple", "modern", "antique", "vintage", "classic",
    "traditional", "wild", "domestic", "outdoor", "indoor", "open", "closed",
    "empty", "full", "running", "sitting", "standing", "lying", "flying",
    "sleeping", "playing", "eating", "drinking", "happy", "sad", "angry",
    "front", "back", "side", "top", "bottom", "left", "right", "upper",
    "lower", "inner", "outer", "main", "primary", "secondary", "central",
    "two", "three", "four", "five", "many", "several", "few", "single",
    "double", "triple",
    "tea", "coffee", "wine", "beer", "soft", "hard", "sweet", "sour",
    "hot", "cold", "warm", "cool", "frozen", "fried", "baked", "roasted",
    "grilled", "boiled", "raw", "cooked",
}

_SYNONYM_GROUPS = {
    "person": ["man", "woman", "people", "kid", "child", "human", "boy", "girl",
               "lady", "guy", "individual", "pedestrian", "baby", "toddler",
               "infant", "adult", "teenager", "teen", "youth", "person's",
               "men", "women", "boys", "girls", "kids", "ladies", "guys",
               "children", "babies", "humans", "pedestrians",
               "father", "mother", "dad", "mom", "parent", "son", "daughter",
               "brother", "sister", "friend", "worker", "player", "rider",
               "skier", "surfer", "skater", "snowboarder", "biker", "cyclist",
               "swimmer", "runner", "passenger", "driver"],
    "puppy": ["dog", "puppy"],
    "kitten": ["cat", "kitten"],
    "cattle": ["cow", "cattle", "ox", "bull"],
    "ox": ["cow", "cattle", "ox", "oxen"],
    "swan": ["duck", "swan", "goose"],
    "goose": ["duck", "goose", "swan", "geese"],
    "hen": ["chicken", "hen", "rooster"],
    "horse": ["horse", "pony"],
    "pony": ["horse", "pony"],
    "fish": ["fish", "salmon", "tuna", "fishes"],
    "couch": ["sofa", "couch"],
    "sofa": ["couch", "sofa"],
    "armchair": ["chair", "armchair"],
    "loveseat": ["sofa", "couch", "loveseat"],
    "ottoman": ["stool", "ottoman", "footstool"],
    "footstool": ["stool", "footstool", "ottoman"],
    "dresser": ["dresser", "chest", "drawer"],
    "wardrobe": ["wardrobe", "closet", "armoire"],
    "tv": ["television", "tv", "television set"],
    "television": ["tv", "television", "television set"],
    "monitor": ["monitor", "screen", "display", "computer monitor"],
    "screen": ["monitor", "screen", "display"],
    "cellphone": ["cell phone", "phone", "mobile phone", "cellular telephone", "smartphone", "iphone", "android"],
    "cell phone": ["cellphone", "phone", "mobile phone", "cellular telephone", "smartphone", "iphone"],
    "smartphone": ["cellphone", "phone", "cell phone", "mobile phone", "iphone"],
    "phone": ["cell phone", "cellphone", "telephone", "smartphone", "cellular telephone"],
    "telephone": ["telephone", "phone", "cell phone", "cellular telephone"],
    "laptop": ["laptop", "laptop computer", "notebook computer", "computer", "macbook"],
    "computer": ["laptop", "computer", "desktop", "pc", "imac", "macbook"],
    "headphones": ["headphones", "earphones", "headset"],
    "earphones": ["earphones", "headphones", "earbuds", "headset"],
    "automobile": ["car", "vehicle", "automobile", "auto", "sedan", "suv"],
    "vehicle": ["car", "vehicle", "automobile", "auto"],
    "motorcycle": ["motorbike", "motorcycle", "bike", "moped", "scooter"],
    "motorbike": ["motorcycle", "motorbike"],
    "bike": ["bicycle", "bike", "cycle"],
    "bicycle": ["bicycle", "bike", "cycle"],
    "lorry": ["truck", "lorry"],
    "van": ["van", "minivan"],
    "subway": ["train", "subway", "metro"],
    "tram": ["train", "tram", "trolley"],
    "airplane": ["airplane", "plane", "aircraft", "jet", "airliner"],
    "plane": ["airplane", "plane", "aircraft", "jet"],
    "ship": ["ship", "boat", "vessel"],
    "boat": ["boat", "ship"],
    "donut": ["doughnut", "donut"],
    "doughnut": ["donut", "doughnut"],
    "hotdog": ["hot dog", "hotdog", "frankfurter", "sausage"],
    "hot dog": ["hotdog", "hot dog", "frankfurter", "sausage"],
    "sausage": ["sausage", "hot dog", "hotdog", "frankfurter", "bratwurst"],
    "fries": ["french fries", "fries", "chips"],
    "burger": ["hamburger", "burger", "cheeseburger"],
    "hamburger": ["hamburger", "burger"],
    "soda": ["soda", "soft drink", "pop"],
    "shrimp": ["shrimp", "prawn"],
    "tshirt": ["t-shirt", "shirt", "tshirt", "tee"],
    "t-shirt": ["t-shirt", "shirt", "tshirt", "tee"],
    "tee": ["t-shirt", "shirt", "tee"],
    "trousers": ["trousers", "pants", "jeans", "slacks"],
    "pants": ["trousers", "pants", "slacks"],
    "sneakers": ["shoe", "sneakers", "trainer", "tennis shoe", "running shoe"],
    "trainers": ["sneakers", "trainers", "shoe"],
    "necktie": ["tie", "necktie"],
    "tie": ["necktie", "tie"],
    "specs": ["glasses", "spectacles", "specs", "eyeglasses"],
    "spectacles": ["glasses", "spectacles", "eyeglasses"],
    "eyeglasses": ["glasses", "eyeglasses", "spectacles"],
    "sunglasses": ["sunglasses", "shades", "glasses"],
    "wineglass": ["wine glass", "wineglass", "glass"],
    "wine glass": ["wineglass", "wine glass"],
    "mug": ["mug", "cup"],
    "tumbler": ["tumbler", "glass", "cup"],
    "platter": ["platter", "plate"],
    "saucer": ["saucer", "plate"],
    "fridge": ["refrigerator", "fridge"],
    "refrigerator": ["refrigerator", "fridge"],
    "microwave": ["microwave", "microwave oven"],
    "stove": ["stove", "oven", "range"],
    "hairdryer": ["hair drier", "hair dryer", "hairdryer", "blow dryer"],
    "hair dryer": ["hair drier", "hair dryer", "hairdryer", "blow dryer"],
    "remote": ["remote", "remote control"],
    "remote control": ["remote", "remote control"],
    "fireplug": ["fireplug", "fire hydrant", "hydrant", "fire plug"],
    "fire hydrant": ["fireplug", "fire hydrant", "hydrant"],
    "train": ["train", "train (railroad vehicle)", "railway", "locomotive"],
    "rucksack": ["backpack", "rucksack", "knapsack"],
    "knapsack": ["backpack", "knapsack", "rucksack"],
    "purse": ["purse", "handbag", "bag"],
    "handbag": ["handbag", "purse", "bag"],
    "bag": ["bag", "handbag", "purse", "tote"],
    "tote": ["tote", "bag", "tote bag"],
    "wallet": ["wallet", "billfold"],
}

_IRREGULAR_PLURALS = {
    "feet": "foot", "mice": "mouse", "leaves": "leaf", "geese": "goose",
    "men": "man", "women": "woman", "children": "child", "teeth": "tooth",
    "oxen": "ox", "wolves": "wolf", "knives": "knife", "lives": "life",
    "loaves": "loaf", "calves": "calf", "shelves": "shelf", "halves": "half",
    "selves": "self", "people": "person", "deer": "deer", "sheep": "sheep",
    "fish": "fish", "fishes": "fish", "cacti": "cactus", "fungi": "fungus",
}


def _normalize(name: str) -> str:
    return name.lower().strip().replace("_", " ").replace("-", " ")


def _evcap_pre_name(caption: str) -> str:
    if not caption:
        return ""
    caption = re.sub(r"([_!,'\"()*#:;~])", " ", caption.lower())
    caption = re.sub(r"\s{2,}", " ", caption)
    return caption.rstrip("\n").strip(" ")


def _variants(name: str) -> List[str]:
    import re as _re
    n = _normalize(name)
    vs = [n]
    m = _re.match(r'^([^(]+?)\s*\(([^)]+)\)\s*$', n)
    if m:
        base = m.group(1).strip()
        paren = m.group(2).strip()
        if base and base not in vs: vs.append(base)
        if paren and paren not in vs: vs.append(paren)
        flat = (base + " " + paren).strip()
        if flat and flat not in vs: vs.append(flat)
    if n in _IRREGULAR_PLURALS:
        sing = _IRREGULAR_PLURALS[n]
        if sing not in vs: vs.append(sing)
    extra = []
    for v in list(vs):
        if v.endswith("s") and len(v) > 2 and not v.endswith("ss"):
            extra.append(v[:-1])
        if not v.endswith("s"):
            extra.append(v + "s")
        if v.endswith("ies") and len(v) > 4:
            extra.append(v[:-3] + "y")
    for e in extra:
        if e not in vs: vs.append(e)
    return vs


def _strip_leading_modifiers(name: str) -> List[str]:
    parts = name.lower().strip().split()
    if len(parts) <= 1:
        return [name.lower().strip()]
    out = [" ".join(parts)]
    cur = parts[:]
    while len(cur) > 1 and cur[0] in _LEADING_MODIFIERS:
        cur = cur[1:]
        out.append(" ".join(cur))
    return out


def build_lvis_lookup(name_values: List[str]) -> Dict[str, int]:
    lookup: Dict[str, int] = {}
    for idx, nm in enumerate(name_values):
        for v in _variants(str(nm)):
            if v not in lookup:
                lookup[v] = idx
    return lookup


def load_lvis_synonyms(lvis_json_path: str) -> Dict[str, List[str]]:
    if not os.path.exists(lvis_json_path):
        print(f"  [LVIS-SYN] ⚠️ 不存在: {lvis_json_path}")
        return {}
    with open(lvis_json_path) as f:
        data = json.load(f)
    syn_map = {}
    for cat in data.get('categories', []):
        name = cat.get('name', '').replace('_', ' ').lower().strip()
        syns = cat.get('synonyms', [])
        if name and syns:
            syn_map[name] = [s.replace('_', ' ').lower().strip() for s in syns]
    print(f"  [LVIS-SYN] {len(syn_map)} 类有官方同义词")
    return syn_map


def build_synonym_lookup(
        lvis_lookup: Dict[str, int],
        lvis_synonyms: Optional[Dict[str, List[str]]] = None,
) -> Dict[str, int]:
    syn_lookup: Dict[str, int] = {}
    for canonical, alternatives in _SYNONYM_GROUPS.items():
        all_names = [canonical] + list(alternatives)
        lvis_idx = None
        for nm in all_names:
            nm_norm = _normalize(nm)
            if nm_norm in lvis_lookup:
                lvis_idx = lvis_lookup[nm_norm]
                break
        if lvis_idx is None:
            continue
        for nm in all_names:
            nm_norm = _normalize(nm)
            if nm_norm and nm_norm not in lvis_lookup and nm_norm not in syn_lookup:
                syn_lookup[nm_norm] = lvis_idx
    if lvis_synonyms:
        for lvis_name, syns in lvis_synonyms.items():
            lvis_idx = lvis_lookup.get(lvis_name.lower().strip())
            if lvis_idx is None:
                continue
            for s in syns:
                s_norm = _normalize(s)
                if s_norm and s_norm not in lvis_lookup and s_norm not in syn_lookup:
                    syn_lookup[s_norm] = lvis_idx
    return syn_lookup


def auto_build_vaw_to_lvis_synonyms(
        vaw_dir: str,
        name_values_str: List[str],
        lvis_lookup: Dict[str, int],
        syn_lookup: Dict[str, int],
) -> Dict[str, int]:
    extra_syn = dict(syn_lookup)
    if not os.path.isdir(vaw_dir):
        return extra_syn
    vaw_names = set()
    for fname in ["train_part1.json", "train_part2.json", "val.json", "test.json"]:
        fp = os.path.join(vaw_dir, fname)
        if not os.path.exists(fp):
            continue
        with open(fp) as f:
            data = json.load(f)
        for entry in data:
            n = _normalize(entry.get("object_name", ""))
            if n: vaw_names.add(n)
    n_added = 0
    n_added_tail = 0
    n_added_head = 0
    n_added_mid = 0
    for vaw in vaw_names:
        if vaw in lvis_lookup or vaw in extra_syn:
            continue
        forms = _strip_leading_modifiers(vaw)
        if any(f in lvis_lookup or f in extra_syn for f in forms):
            continue
        words = vaw.split()
        if len(words) < 2:
            continue
        matched_to = None
        match_kind = ""
        for k in range(1, len(words)):
            tail = " ".join(words[-k:])
            for v in _variants(tail):
                if v in lvis_lookup:
                    matched_to = lvis_lookup[v]
                    match_kind = "tail"
                    break
            if matched_to is not None: break
        if matched_to is None:
            for k in range(1, len(words)):
                head = " ".join(words[:k])
                for v in _variants(head):
                    if v in lvis_lookup:
                        matched_to = lvis_lookup[v]
                        match_kind = "head"
                        break
                if matched_to is not None: break
        if matched_to is None:
            for w in words:
                if len(w) < 4: continue
                for v in _variants(w):
                    if v in lvis_lookup:
                        matched_to = lvis_lookup[v]
                        match_kind = "mid"
                        break
                if matched_to is not None: break
        if matched_to is not None:
            extra_syn[vaw] = matched_to
            n_added += 1
            if match_kind == "tail":
                n_added_tail += 1
            elif match_kind == "head":
                n_added_head += 1
            else:
                n_added_mid += 1
    print(f"  [VAW→LVIS auto] 额外补 {n_added} 条 (尾词:{n_added_tail}, 首词:{n_added_head}, 中间:{n_added_mid})")
    return extra_syn


def _extract_prepositional_heads(name: str) -> List[str]:
    n = name.lower().strip()
    candidates = []
    m = re.match(r'^([\w\s]+?)\s+of\s+([\w\s]+?)$', n)
    if m:
        x = m.group(1).strip();
        y = m.group(2).strip()
        if y: candidates.append(y)
        if x: candidates.append(x)
    m = re.match(r"^([\w\s]+?)\s*'s\s+([\w\s]+?)$", n)
    if m:
        x = m.group(1).strip();
        y = m.group(2).strip()
        if y: candidates.append(y)
        if x: candidates.append(x)
    m = re.match(r'^([\w\s]+?)\s+with\s+([\w\s]+?)$', n)
    if m:
        x = m.group(1).strip()
        if x: candidates.append(x)
    m = re.match(r'^([\w\s]+?)\s+in\s+([\w\s]+?)$', n)
    if m:
        x = m.group(1).strip();
        y = m.group(2).strip()
        if x: candidates.append(x)
        if y: candidates.append(y)
    seen = set()
    return [c for c in candidates if c and not (c in seen or seen.add(c))]


def match_to_lvis(
        name: str,
        lvis_lookup: Dict[str, int],
        syn_lookup: Dict[str, int],
        multi: bool = True,
) -> Tuple[List[int], str]:
    name_norm = _normalize(name)
    if not name_norm:
        return [], "empty"
    matched = []
    reason = "no_match"
    for v in _variants(name_norm):
        if v in lvis_lookup:
            idx = lvis_lookup[v]
            if idx not in matched:
                matched.append(idx)
                reason = "exact" if v == name_norm else "plural"
                if not multi:
                    return matched, reason
                break
    forms = _strip_leading_modifiers(name_norm)
    for f in forms[1:]:
        for v in _variants(f):
            if v in lvis_lookup:
                idx = lvis_lookup[v]
                if idx not in matched:
                    matched.append(idx)
                    if reason == "no_match":
                        reason = "modifier"
                    if not multi:
                        return matched, reason
                    break
    if name_norm in syn_lookup:
        idx = syn_lookup[name_norm]
        if idx not in matched:
            matched.append(idx)
            if reason == "no_match":
                reason = "synonym"
            if not multi:
                return matched, reason
    for f in forms[1:]:
        if f in syn_lookup:
            idx = syn_lookup[f]
            if idx not in matched:
                matched.append(idx)
                if reason == "no_match":
                    reason = "modifier+synonym"
                if not multi:
                    return matched, reason
    prep_heads = _extract_prepositional_heads(name_norm)
    for ph in prep_heads:
        for v in _variants(ph):
            if v in lvis_lookup:
                idx = lvis_lookup[v]
                if idx not in matched:
                    matched.append(idx)
                    if reason == "no_match":
                        reason = "prep"
                    if not multi:
                        return matched, reason
                    break
        if ph in syn_lookup:
            idx = syn_lookup[ph]
            if idx not in matched:
                matched.append(idx)
                if reason == "no_match":
                    reason = "prep+synonym"
                if not multi:
                    return matched, reason
    return matched, reason


# ════════════════════════════════════════════════════════════════════
# select_attrs_adaptive (跟 v6.9.1 完全一致)
# ════════════════════════════════════════════════════════════════════
def select_attrs_adaptive(
        obj2dim2attrs: Dict[str, Dict[str, List[Tuple[str, int]]]],
        obj_total_count: Dict[str, int],
        min_ratio: float = MIN_RATIO,
        min_count: int = MIN_COUNT,
        max_per_dim: int = MAX_PER_DIM,
        max_per_obj: int = TOPK_ATTRS,
        min_dims_per_obj: int = 1,
        source_tag: str = "?",
) -> List[Tuple[str, str, str, int]]:
    final_triples = []
    n_dropped_low_dim = 0
    n_passed = 0

    for obj in sorted(obj2dim2attrs.keys()):
        obj_tot = max(obj_total_count.get(obj, 1), 1)
        candidates = []
        for dim, attr_list in obj2dim2attrs[obj].items():
            top = sorted(attr_list, key=lambda x: -x[1])[:max_per_dim]
            for av, n in top:
                ratio = n / obj_tot
                if n >= min_count and ratio >= min_ratio:
                    candidates.append((dim, av, n, ratio))
        candidates.sort(key=lambda x: -x[3])
        candidates = candidates[:max_per_obj]

        unique_dims = set(c[0] for c in candidates)
        if len(unique_dims) < min_dims_per_obj:
            n_dropped_low_dim += 1
            continue

        n_passed += 1
        for dim, av, n, ratio in candidates:
            final_triples.append((obj, dim, av, n))

    if n_dropped_low_dim > 0 or min_dims_per_obj > 1:
        print(f"    [{source_tag} dim 多样性] "
              f"通过 {n_passed} obj, 丢弃 {n_dropped_low_dim} obj "
              f"(dim<{min_dims_per_obj})")
    return final_triples


def ensure_eva_vit_g(path: str, url: str = EVA_VIT_G_URL) -> str:
    if os.path.exists(path):
        return path
    print(f"  [eva_vit_g] 下载: {url} → {path}")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = path + ".tmp"

    def _progress(bn, bs, ts):
        if ts > 0:
            pct = min(100.0, 100.0 * bn * bs / ts)
            print(f"\r  [eva_vit_g] {bn * bs / 1e6:.1f}/{ts / 1e6:.1f} MB ({pct:.1f}%)", end='', flush=True)

    try:
        urllib.request.urlretrieve(url, tmp_path, reporthook=_progress)
        print();
        os.rename(tmp_path, path)
    except Exception as e:
        if os.path.exists(tmp_path): os.remove(tmp_path)
        raise RuntimeError(f"eva_vit_g 下载失败: {e}")
    return path


def find_vg_image(image_id, vg_root):
    fname = f"{image_id}.jpg"
    candidates = [
        os.path.join(vg_root, "VG_100K", fname),
        os.path.join(vg_root, "VG_100K_2", fname),
        os.path.join(vg_root, fname),
        os.path.join(vg_root, "images", fname),
        os.path.join(vg_root, "images", "VG_100K", fname),
        os.path.join(vg_root, "images", "VG_100K_2", fname),
        os.path.join(os.path.dirname(vg_root), "VG_100K", fname),
        os.path.join(os.path.dirname(vg_root), "VG_100K_2", fname),
    ]
    for p in candidates:
        if os.path.exists(p): return p
    return None


def probe_vg_root(test_ids, candidate_roots):
    best_root = candidate_roots[0];
    best_count = 0
    for root in candidate_roots:
        cnt = sum(1 for iid in test_ids if find_vg_image(iid, root) is not None)
        print(f"    探测: {root}  → {cnt}/{len(test_ids)}")
        if cnt > best_count:
            best_count = cnt;
            best_root = root
    return best_root, best_count


def find_coco_image(image_id, coco_roots, file_name=None):
    if isinstance(coco_roots, (list, tuple)):
        input_roots = list(coco_roots)
    else:
        input_roots = [coco_roots]
    roots = []
    for r in input_roots:
        if r and r not in roots:
            roots.append(r)
        for sub in ("train2014", "val2014", "train2017", "val2017"):
            sp = os.path.join(r, sub)
            if os.path.isdir(sp) and sp not in roots:
                roots.append(sp)
    img_id_int = int(image_id)
    fname_candidates = []
    if file_name:
        fname_candidates.append(file_name)
        fname_candidates.append(os.path.basename(file_name))
    fname_candidates += [
        f"{img_id_int:012d}.jpg",
        f"COCO_train2014_{img_id_int:012d}.jpg",
        f"COCO_val2014_{img_id_int:012d}.jpg",
        f"{image_id}.jpg",
    ]
    seen = set()
    fname_candidates = [f for f in fname_candidates if not (f in seen or seen.add(f))]
    for root in roots:
        for fmt in fname_candidates:
            p = os.path.join(root, fmt)
            if os.path.exists(p):
                return p
    return None


def crop_bbox(img, bbox, margin=BBOX_MARGIN):
    W, H = img.size
    x, y, w, h = bbox
    x1 = max(0, x - w * margin);
    y1 = max(0, y - h * margin)
    x2 = min(W, x + w * (1 + margin));
    y2 = min(H, y + h * (1 + margin))
    if x2 - x1 < 2 or y2 - y1 < 2:
        return img
    return img.crop((x1, y1, x2, y2))


# ════════════════════════════════════════════════════════════════════
# BLIP2Encoder (跟 v6.9.1 一致)
# ════════════════════════════════════════════════════════════════════
class BLIP2Encoder:
    def __init__(self, eva_clip_path, blip2_ckpt, device="cuda"):
        self.device = device
        self.feature_dim = 768
        print(f"\n{'=' * 60}")
        print(f"  加载 BLIP-2 编码器")
        print(f"{'=' * 60}")
        self._load_visual(eva_clip_path)
        self._load_qformer(blip2_ckpt)
        print(f"  编码器就绪 ✅\n")

    def _load_visual(self, path):
        print(f"  [1] 加载 LAVIS EVA-ViT-g/14 ...")
        ensure_eva_vit_g(path)
        try:
            from lavis_eva_vit import VisionTransformer, interpolate_pos_embed
            from functools import partial
        except ImportError as e:
            raise RuntimeError(f"导入 lavis_eva_vit 失败: {e}")
        model = VisionTransformer(
            img_size=224, patch_size=14, use_mean_pooling=False,
            embed_dim=1408, depth=39, num_heads=1408 // 88,
            mlp_ratio=4.3637, qkv_bias=True, drop_path_rate=0.0,
            norm_layer=partial(nn.LayerNorm, eps=1e-6), use_checkpoint=False,
        )
        state_dict = torch.load(path, map_location='cpu', weights_only=False)
        if isinstance(state_dict, dict) and 'model' in state_dict:
            state_dict = state_dict['model']
        try:
            interpolate_pos_embed(model, state_dict)
        except Exception:
            pass
        miss, unexp = model.load_state_dict(state_dict, strict=False)
        print(f"      load: missing={len(miss)} unexpected={len(unexp)}")
        model = model.to(device=self.device, dtype=torch.bfloat16).eval().requires_grad_(False)
        from torchvision import transforms
        from torchvision.transforms.functional import InterpolationMode
        self.preprocess = transforms.Compose([
            transforms.Resize((224, 224), interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.48145466, 0.4578275, 0.40821073),
                std=(0.26862954, 0.26130258, 0.27577711),
            ),
        ])
        self.visual_encoder = model
        print(f"      ✅ EVA-ViT-g 加载完成")

    def _load_qformer(self, ckpt_path):
        print(f"  [2] 加载 LAVISQFormerForRetrieval ...")
        try:
            from caption_model.retrieval_qformer import LAVISQFormerForRetrieval
        except ImportError as e:
            raise RuntimeError(f"导入 caption_model 失败: {e}")
        qformer = LAVISQFormerForRetrieval(
            num_query_token=NUM_QUERY, vision_width=1408, hidden_size=768,
            num_hidden_layers=12, num_attention_heads=12,
            intermediate_size=3072, cross_attention_freq=2,
        )
        qformer.load_blip2_weights(ckpt_path)
        if not qformer._loaded:
            raise RuntimeError(f"LAVISQFormerForRetrieval 加载失败: {ckpt_path}")
        qformer = qformer.to(self.device).eval()
        for p in qformer.parameters(): p.requires_grad = False
        self.qformer = qformer
        print(f"      ✅ Q-Former 加载完成")

    @torch.no_grad()
    def encode_images(self, images, batch_size=BATCH_SIZE):
        all_vecs = []
        for i in range(0, len(images), batch_size):
            batch = images[i: i + batch_size]
            tensors = torch.stack([self.preprocess(im.convert("RGB")) for im in batch]).to(self.device)
            _amp = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            with torch.amp.autocast(device_type="cuda", dtype=_amp):
                patch = self.visual_encoder(tensors)
            last = self.qformer(patch.float())
            feat = F.normalize(last.mean(dim=1), dim=-1)
            all_vecs.append(feat.cpu().numpy())
        return np.vstack(all_vecs).astype(np.float32)


# ════════════════════════════════════════════════════════════════════
# VAW 收集器 (跟 v6.9.1 完全一致)
# ════════════════════════════════════════════════════════════════════
def collect_vaw_triples(
        vaw_dir: str,
        name_values_str: List[str],
        lvis_lookup: Dict[str, int],
        syn_lookup: Dict[str, int],
        unified_attr2dim: Dict[str, str],
        debug: bool = False,
) -> Tuple[Dict, Dict[str, int], Counter]:
    print(f"\n{'-' * 50}")
    print(f"  [VAW] 收集 (v6.9.2: unified attr→dim 表已过滤)")
    print(f"{'-' * 50}")

    triple_to_images = defaultdict(list)
    obj_total_count = defaultdict(int)
    total = n_skip = n_no_match = multi_count = n_no_dim = 0
    match_stats = Counter()
    vaw_to_lvis_cache: Dict[str, List[int]] = {}

    for fname in ["train_part1.json", "train_part2.json", "val.json", "test.json"]:
        fp = os.path.join(vaw_dir, fname)
        if not os.path.exists(fp):
            continue
        print(f"  读: {fname}")
        with open(fp) as f:
            data = json.load(f)
        for entry in data:
            total += 1
            raw_obj = _normalize(entry.get("object_name", ""))
            bbox = entry.get("instance_bbox", None)
            img_id = entry.get("image_id", None)
            if not raw_obj or bbox is None or img_id is None:
                n_skip += 1;
                continue
            if raw_obj in vaw_to_lvis_cache:
                lvis_idxs = vaw_to_lvis_cache[raw_obj]
            else:
                lvis_idxs, reason = match_to_lvis(raw_obj, lvis_lookup, syn_lookup, multi=True)
                vaw_to_lvis_cache[raw_obj] = lvis_idxs
                match_stats[reason] += 1
            if not lvis_idxs:
                n_no_match += 1
                continue
            if len(lvis_idxs) > 1:
                multi_count += 1
            for lvis_idx in lvis_idxs:
                canonical_obj = _normalize(name_values_str[lvis_idx])
                obj_total_count[canonical_obj] += 1
                for attr in entry.get("positive_attributes", []):
                    av = attr.lower().strip()
                    if av in ATTR_BLACKLIST:
                        continue
                    d = unified_attr2dim.get(av, "")
                    if not d:
                        n_no_dim += 1
                        continue
                    triple_to_images[(canonical_obj, d, av)].append((img_id, bbox))

    print(f"\n  [VAW 统计]")
    print(f"    总 entries:               {total:,}")
    print(f"    无 LVIS 匹配:             {n_no_match:,}")
    print(f"    attr 词不在过滤后 dim 表: {n_no_dim:,}")
    print(f"    收集 (obj,dim,av):        {len(triple_to_images):,}")
    print(f"    覆盖 LVIS canonical:      {len(obj_total_count)}")
    dim_distribution = Counter(t[1] for t in triple_to_images.keys())
    print(f"    实际入库 dim 数 (VAW):    {len(dim_distribution)}")
    print(f"    实际 dim 分布 (前 15):")
    for d, c in dim_distribution.most_common(15):
        print(f"      {d:25s}: {c}")
    return triple_to_images, obj_total_count, match_stats


# ════════════════════════════════════════════════════════════════════
# PACO 收集器 (跟 v6.9.1 完全一致)
# ════════════════════════════════════════════════════════════════════
def collect_paco_triples(
        paco_json: str,
        name_values_str: List[str],
        lvis_lookup: Dict[str, int],
        syn_lookup: Dict[str, int],
        dim_whitelist: Optional[Set[str]] = None,
        caption_freq: Optional[Counter] = None,
        freq_threshold: int = 0,
) -> Tuple[Dict, Dict[str, int], Dict, Dict]:
    print(f"\n{'-' * 50}")
    print(f"  [PACO] 收集 (v6.9.2: 加 dim 白名单 + 词频过滤)")
    print(f"{'-' * 50}")
    if not os.path.exists(paco_json):
        print(f"  ❌ PACO json 不存在: {paco_json}")
        return {}, {}, {}, {}
    with open(paco_json) as f:
        paco = json.load(f)
    print(f"    images: {len(paco.get('images', [])):,}, annotations: {len(paco.get('annotations', [])):,}")
    print(f"    categories: {len(paco.get('categories', [])):,}, attributes: {len(paco.get('attributes', [])):,}")

    def _paco_name_normalize(raw: str) -> List[str]:
        s = raw.replace('_', ' ').lower().strip()
        candidates = [s]
        m = re.match(r'^([^(]+?)\s*\(([^)]+)\)\s*$', s)
        if m:
            candidates.extend([m.group(1).strip(), m.group(2).strip()])
        words = s.replace('(', '').replace(')', '').split()
        if len(words) >= 2:
            candidates.append(' '.join(words[:-1]))
            candidates.append(words[-1])
        seen = set()
        return [c for c in candidates if c and not (c in seen or seen.add(c))]

    cat_id_to_name = {};
    cat_id_to_aliases = {}
    for cat in paco.get('categories', []):
        candidates = _paco_name_normalize(cat['name'])
        if ':' in candidates[0]:
            continue
        cat_id_to_name[cat['id']] = candidates[0]
        cat_id_to_aliases[cat['id']] = candidates
    obj_cat_ids = set(cat_id_to_name.keys())

    attr_type_to_idxs = paco.get('attr_type_to_attr_idxs', {})
    attr_id_to_type = {}
    for a_type, idxs in attr_type_to_idxs.items():
        for aid in idxs:
            attr_id_to_type[aid] = a_type.lower().strip()
    attr_id_to_info = {}
    for attr in paco.get('attributes', []):
        aid = attr['id']
        a_name = attr['name'].lower().strip().replace('_', ' ')
        a_type = attr_id_to_type.get(aid, "")
        attr_id_to_info[aid] = (a_type, a_name)

    img_id_to_file = {im['id']: im['file_name'] for im in paco['images']}

    triple_to_boxes = defaultdict(list)
    obj_total = defaultdict(int)
    n_no_lvis = n_part = n_no_attrs = n_skipped_dim = n_skipped_wl = n_skipped_freq = 0
    paco_to_lvis_cache: Dict[str, int] = {}

    for ann in paco['annotations']:
        cid = ann.get('category_id')
        if cid not in obj_cat_ids:
            n_part += 1;
            continue
        obj_name = cat_id_to_name[cid]
        bbox = ann.get('bbox');
        img_id = ann.get('image_id')
        if not bbox or img_id is None: continue
        if obj_name not in paco_to_lvis_cache:
            lvis_idx = -1
            for cand in cat_id_to_aliases[cid]:
                lvis_idxs, _ = match_to_lvis(cand, lvis_lookup, syn_lookup, multi=False)
                if lvis_idxs:
                    lvis_idx = lvis_idxs[0];
                    break
            paco_to_lvis_cache[obj_name] = lvis_idx
        lvis_idx = paco_to_lvis_cache[obj_name]
        if lvis_idx < 0:
            n_no_lvis += 1;
            continue
        canonical_obj = _normalize(name_values_str[lvis_idx])
        obj_total[canonical_obj] += 1
        attr_ids = list(ann.get('attribute_ids', []) or [])
        dom_color_ids = list(ann.get('dom_color_ids', []) or [])
        all_attr_ids = list(set(attr_ids + dom_color_ids))
        if not all_attr_ids:
            n_no_attrs += 1;
            continue
        for aid in all_attr_ids:
            info = attr_id_to_info.get(aid)
            if not info: continue
            a_type, a_name = info
            if a_name in ATTR_BLACKLIST: continue
            dim = PACO_TYPE_TO_DIM.get(a_type, a_type if a_type else "")
            if not dim:
                n_skipped_dim += 1
                continue

            if dim_whitelist is not None and dim not in dim_whitelist:
                n_skipped_wl += 1
                continue

            if caption_freq is not None and freq_threshold > 0:
                if not all(caption_freq.get(w, 0) >= freq_threshold
                           for w in a_name.split()):
                    n_skipped_freq += 1
                    continue

            triple_to_boxes[(canonical_obj, dim, a_name)].append((img_id, bbox))

    print(f"\n  [PACO 统计]")
    print(f"    跳过 part:                {n_part:,}")
    print(f"    obj 无 LVIS 匹配:         {n_no_lvis:,}")
    print(f"    跳过 dim 为空:            {n_skipped_dim:,}")
    if dim_whitelist is not None:
        print(f"    [v6.9] dim 不在白名单:    {n_skipped_wl:,}")
    if caption_freq is not None and freq_threshold > 0:
        print(f"    [v6.9] attr 词频不够:     {n_skipped_freq:,}")
    print(f"    收集 (obj,dim,av):        {len(triple_to_boxes):,}")
    print(f"    覆盖 LVIS canonical:      {len(obj_total)}")
    return triple_to_boxes, obj_total, img_id_to_file, attr_id_to_info


# ════════════════════════════════════════════════════════════════════
# VG 收集器 (跟 v6.9.1 完全一致)
# ════════════════════════════════════════════════════════════════════
def collect_vg_triples(
        attributes_json: str,
        image_data_json: Optional[str],
        name_values_str: List[str],
        lvis_lookup: Dict[str, int],
        syn_lookup: Dict[str, int],
        unified_attr2dim: Dict[str, str],
        max_images_per_attr: int = 80,
        min_box_per_attr: int = 1,
        debug: bool = False,
) -> Tuple[Dict, Dict[str, int], Dict, Counter]:
    print(f"\n{'-' * 50}")
    print(f"  [VG] 收集 (v6.9.2: 严格表 + min_box={min_box_per_attr}, max_box={max_images_per_attr})")
    print(f"{'-' * 50}")
    if not os.path.exists(attributes_json):
        print(f"  ❌ VG attributes.json 不存在: {attributes_json}")
        return {}, {}, {}, Counter()

    vg_image_meta = {}
    if image_data_json and os.path.exists(image_data_json):
        print(f"  读 image_data.json ...")
        with open(image_data_json) as f:
            img_data = json.load(f)
        for im in img_data:
            iid = im.get('image_id') or im.get('id')
            if iid is None: continue
            vg_image_meta[iid] = {
                'width': im.get('width'),
                'height': im.get('height'),
                'coco_id': im.get('coco_id'),
            }
        print(f"    → {len(vg_image_meta):,} VG 图像元信息")

    print(f"  读 attributes.json (~440 MB, 可能要 30-60 秒) ...")
    with open(attributes_json) as f:
        vg_data = json.load(f)
    print(f"    → {len(vg_data):,} 张图")

    triple_to_images = defaultdict(list)
    obj_total_count = defaultdict(int)
    vg_to_lvis_cache: Dict[str, List[int]] = {}
    match_stats = Counter()
    n_images = n_objs = n_no_attrs = n_no_match = n_not_in_table = n_collected = 0
    multi_count = 0

    for img_entry in vg_data:
        n_images += 1
        img_id = img_entry.get('image_id') or img_entry.get('id')
        if img_id is None: continue
        for obj in img_entry.get('attributes', []):
            n_objs += 1
            obj_attrs = obj.get('attributes', []) or []
            if not obj_attrs:
                n_no_attrs += 1;
                continue
            obj_names = obj.get('names', []) or []
            if not obj_names: continue
            x = obj.get('x');
            y = obj.get('y')
            w = obj.get('w');
            h = obj.get('h')
            if any(v is None for v in (x, y, w, h)): continue
            if w < 5 or h < 5: continue
            bbox = [x, y, w, h]
            for raw_obj_name in obj_names[:2]:
                raw_obj = _normalize(raw_obj_name)
                if not raw_obj: continue
                if raw_obj in vg_to_lvis_cache:
                    lvis_idxs = vg_to_lvis_cache[raw_obj]
                else:
                    lvis_idxs, reason = match_to_lvis(raw_obj, lvis_lookup, syn_lookup, multi=True)
                    vg_to_lvis_cache[raw_obj] = lvis_idxs
                    match_stats[reason] += 1
                if not lvis_idxs:
                    n_no_match += 1
                    continue
                if len(lvis_idxs) > 1:
                    multi_count += 1
                for lvis_idx in lvis_idxs:
                    canonical_obj = _normalize(name_values_str[lvis_idx])
                    obj_total_count[canonical_obj] += 1
                    for attr in obj_attrs:
                        av = _normalize(attr)
                        if not av:
                            continue
                        if av in ATTR_BLACKLIST:
                            continue
                        d = unified_attr2dim.get(av, "")
                        if not d:
                            n_not_in_table += 1
                            continue
                        triples_list = triple_to_images[(canonical_obj, d, av)]
                        if len(triples_list) < max_images_per_attr:
                            triples_list.append((img_id, bbox))
                            n_collected += 1
                break

    n_before_filter = len(triple_to_images)
    if min_box_per_attr > 1:
        triple_to_images = {
            k: v for k, v in triple_to_images.items()
            if len(v) >= min_box_per_attr
        }
        n_dropped = n_before_filter - len(triple_to_images)
        print(f"\n  [v6.9.1 box 数后置过滤]")
        print(f"    {n_before_filter:,} (obj,dim,av) → {len(triple_to_images):,} "
              f"(-{n_dropped:,}, 丢弃 box<{min_box_per_attr})")
        valid_objs = set(k[0] for k in triple_to_images.keys())
        obj_total_count = {o: c for o, c in obj_total_count.items() if o in valid_objs}

    print(f"\n  [VG 统计]")
    print(f"    总图数:                {n_images:,}")
    print(f"    总对象数:              {n_objs:,}")
    print(f"    无 LVIS 匹配:          {n_no_match:,}")
    print(f"    attr 词不在 VG严格表:   {n_not_in_table:,}")
    print(f"    收集 (obj,dim,av):     {len(triple_to_images):,}")
    print(f"    bbox 总数 (含上限):     {n_collected:,}")
    print(f"    覆盖 LVIS canonical:   {len(obj_total_count)}")
    actual_vg_dims = Counter(t[1] for t in triple_to_images.keys())
    print(f"    实际入库 dim 数 (VG):  {len(actual_vg_dims)}")
    return triple_to_images, obj_total_count, vg_image_meta, match_stats


def _dedupe_boxes(boxes: List[Tuple[int, list]]) -> List[Tuple[int, list]]:
    seen = set()
    out = []
    for img_id, bbox in boxes:
        if not bbox or len(bbox) < 4:
            continue
        try:
            key = (int(img_id),
                   round(float(bbox[0]), 1),
                   round(float(bbox[1]), 1),
                   round(float(bbox[2]), 1),
                   round(float(bbox[3]), 1))
        except (ValueError, TypeError):
            continue
        if key in seen:
            continue
        seen.add(key)
        out.append((img_id, bbox))
    return out


def extract_prototypes(
        final_triples, triple_to_boxes, encoder, img_finder,
        max_per_attr: int = MAX_PER_ATTR,
        margin: float = BBOX_MARGIN,
        batch: int = BATCH_SIZE,
        source_tag: str = "?",
):
    attr_vectors = []
    all_meta = []
    n_no_img = n_err = 0
    n_total_boxes = 0
    n_dedup_boxes = 0
    for tri_idx, (obj, dim, av, n_total) in enumerate(final_triples):
        raw_boxes = triple_to_boxes[(obj, dim, av)]
        deduped = _dedupe_boxes(raw_boxes)
        n_total_boxes += len(raw_boxes)
        n_dedup_boxes += len(deduped)
        boxes = deduped[:max_per_attr]
        crops = []
        for img_id, bbox in boxes:
            img_path = img_finder(img_id)
            if not img_path:
                continue
            try:
                img = Image.open(img_path).convert("RGB")
                crops.append(crop_bbox(img, bbox, margin))
            except Exception:
                n_err += 1;
                continue
        if not crops:
            n_no_img += 1;
            continue
        vecs = encoder.encode_images(crops, batch_size=batch)
        proto = vecs.mean(axis=0)
        proto = (proto / (np.linalg.norm(proto) + 1e-8)).astype(np.float32)
        attr_vectors.append(proto)
        all_meta.append({
            "obj_name": obj, "dim": dim, "attr_value": av,
            "n_images": len(crops), "n_total": n_total,
            "text": f"a {av} {obj}",
            "source": source_tag,
        })
        if (tri_idx + 1) % 500 == 0 or tri_idx == len(final_triples) - 1:
            print(f"    [{len(all_meta):>5}/{len(final_triples)}] {obj:20s} {dim:20s} {av:15s} ({len(crops)}图)")
    dedup_pct = 100.0 * (n_total_boxes - n_dedup_boxes) / max(n_total_boxes, 1)
    print(f"  [{source_tag}] 提取完成: {len(all_meta)}/{len(final_triples)}, 无图: {n_no_img}, 错: {n_err}")
    print(f"  [{source_tag}] bbox 去重: {n_total_boxes:,} → {n_dedup_boxes:,} (-{dedup_pct:.1f}%)")
    return attr_vectors, all_meta


# ════════════════════════════════════════════════════════════════════
# 主流程
# ════════════════════════════════════════════════════════════════════
def main():
    p = argparse.ArgumentParser(description="构建属性记忆库 v6.9.2 (适配 v3 classify)")
    p.add_argument("--use_paco", action="store_true")
    p.add_argument("--use_vaw", action="store_true")
    p.add_argument("--use_vg", action="store_true")
    p.add_argument("--paco_json", default=PACO_JSON)
    p.add_argument("--coco_images", default=COCO_IMAGES, nargs='+')
    p.add_argument("--vaw_dir", default=VAW_DIR)
    p.add_argument("--vg_images", default=VG_IMAGES_DIR)
    p.add_argument("--lvis_json", default=None)
    p.add_argument("--vg_attributes_json", default=VG_ATTRIBUTES_JSON)
    p.add_argument("--vg_image_data_json", default=VG_IMAGE_DATA_JSON)
    p.add_argument("--vg_attr_types_json", default=VG_ATTR_TYPES_JSON)
    p.add_argument("--vg_max_per_attr", type=int, default=VG_MAX_BOX_PER_ATTR)
    p.add_argument("--name_pkl", default=NAME_PKL_PATH)
    p.add_argument("--eva_clip", default=EVA_CLIP_PATH)
    p.add_argument("--blip2_ckpt", default=BLIP2_CKPT)
    p.add_argument("--output", default=OUTPUT_DIR)
    p.add_argument("--topk", type=int, default=TOPK_ATTRS)
    p.add_argument("--min_count", type=int, default=MIN_COUNT)
    p.add_argument("--min_ratio", type=float, default=MIN_RATIO)
    p.add_argument("--max_per_dim", type=int, default=MAX_PER_DIM)
    p.add_argument("--max_per_attr", type=int, default=MAX_PER_ATTR)
    p.add_argument("--min_images", type=int, default=MIN_IMAGES)
    p.add_argument("--batch", type=int, default=BATCH_SIZE)
    p.add_argument("--margin", type=float, default=BBOX_MARGIN)
    p.add_argument("--debug_match", action="store_true")

    p.add_argument("--caption_freq_threshold", type=int, default=CAPTION_FREQ_THRESHOLD)
    p.add_argument("--coco_caption_json", default=COCO_CAPTION_JSON)
    p.add_argument("--min_dims_per_obj", type=int, default=MIN_DIMS_PER_OBJ)
    p.add_argument("--no_dim_whitelist", action="store_true")

    p.add_argument("--vg_caption_freq_threshold", type=int, default=VG_CAPTION_FREQ_THRESHOLD)
    p.add_argument("--vg_min_box_per_attr", type=int, default=VG_MIN_BOX_PER_ATTR)
    p.add_argument("--vg_no_restrict_to_vaw_dims", action="store_true")

    args = p.parse_args()

    if not args.use_paco and not args.use_vaw and not args.use_vg:
        print("  [v6.9.2 默认] PACO + VAW + VG 全部启用")
        args.use_paco = True
        args.use_vaw = True
        args.use_vg = True

    os.makedirs(args.output, exist_ok=True)

    SEP = "=" * 70
    print(f"\n{SEP}\n  属性记忆库构建 v6.9.2 (适配 v3 classify)\n{SEP}")
    print(f"\n  ★ v6.9.2 核心改动 (相对 v6.9.1):")
    print(f"    1) 重写 CAPTION_FRIENDLY_DIMS 跟 v3 输出严格对齐")
    print(f"       - 修 v6.9.1 的 bug: 'face_expression' → 'face expression'")
    print(f"       - 修 v6.9.1 的 bug: 'hair_color'      → 'hair color'")
    print(f"       - 修 v6.9.1 的 bug: 'hair_length'     → 'hair length'")
    print(f"    2) 收纳 v3 新增 caption-friendly dim:")
    print(f"       temperature/taste/wetness/lighting/season/emotion/...")
    print(f"    3) 主动砍掉 34 个低 caption 价值的 dim")
    print(f"    白名单大小: {len(CAPTION_FRIENDLY_DIMS)} 个 dim")
    print(f"\n  数据源:  PACO={args.use_paco}  VAW={args.use_vaw}  VG={args.use_vg}")

    # ── STEP 0: 加载 attribute_types ──
    print(f"\n{'-' * 50}\nSTEP 0 | 加载 attribute_types\n{'-' * 50}")
    vaw_types = load_vaw_attr_types(args.vaw_dir) if args.use_vaw else {}
    vg_types = load_vg_attr_types(args.vg_attr_types_json) if args.use_vg else {}
    if not vaw_types and not vg_types:
        print(f"  ❌ VAW + VG attribute_types 都没加载到, 退出");
        return
    unified_attr2dim, unified_dim2attrs = merge_attr_types(
        vaw_types, vg_types, vaw_priority=True)

    # ── STEP 0.5: v6.9.2 双重过滤 ──
    print(f"\n{'-' * 50}\nSTEP 0.5 | v6.9.2 双重过滤 (白名单 + 词频)\n{'-' * 50}")
    if args.no_dim_whitelist:
        print(f"  ⚠️ --no_dim_whitelist 启用, 跳过白名单过滤 (debug)")
        caption_freq = None
        active_dim_whitelist = None
    else:
        caption_freq = build_caption_word_freq(args.coco_caption_json) \
            if args.caption_freq_threshold > 0 else None
        active_dim_whitelist = CAPTION_FRIENDLY_DIMS
        unified_attr2dim, unified_dim2attrs = filter_unified_by_whitelist_and_freq(
            unified_attr2dim, unified_dim2attrs,
            dim_whitelist=active_dim_whitelist,
            caption_freq=caption_freq,
            freq_threshold=args.caption_freq_threshold,
        )

    # ── STEP 0.6: VG 严格表 ──
    if args.use_vg:
        print(f"\n{'-' * 50}\nSTEP 0.6 | v6.9.1 VG 严格表\n{'-' * 50}")
        vaw_actual_dims = set(unified_dim2attrs.keys())
        vg_attr2dim = build_vg_strict_attr2dim(
            unified_attr2dim,
            caption_freq=caption_freq,
            vg_freq_threshold=args.vg_caption_freq_threshold,
            vaw_dims=vaw_actual_dims,
            restrict_to_vaw_dims=(not args.vg_no_restrict_to_vaw_dims),
        )
    else:
        vg_attr2dim = {}

    # ── STEP 1: 读 LVIS pkl ──
    print(f"\n{'-' * 50}\nSTEP 1 | 读 ext_memory_lvis.pkl\n{'-' * 50}")
    with open(args.name_pkl, 'rb') as f:
        nm = pickle.load(f)
    if isinstance(nm, dict):
        name_keys = nm.get('keys', nm.get('key'))
        name_values = nm.get('values', nm.get('value'))
    else:
        name_keys, name_values = nm
    if isinstance(name_keys, torch.Tensor): name_keys = name_keys.cpu().float().numpy()
    if isinstance(name_values, torch.Tensor): name_values = name_values.cpu().tolist()
    if isinstance(name_values, np.ndarray): name_values = name_values.tolist()
    name_keys = np.array(name_keys, dtype=np.float32)
    name_dim = name_keys.shape[-1]
    name_values_str = [str(v) for v in name_values]
    unique_lvis_names = sorted(set(_normalize(n) for n in name_values_str))
    print(f"  LVIS pkl: {name_keys.shape}, {len(unique_lvis_names)} 唯一类")

    lvis_lookup = build_lvis_lookup(name_values_str)
    lvis_synonyms = load_lvis_synonyms(args.lvis_json) if args.lvis_json else {}
    syn_lookup = build_synonym_lookup(lvis_lookup, lvis_synonyms)
    print(f"  LVIS lookup: {len(lvis_lookup)}, 同义词 lookup: {len(syn_lookup)}")
    if args.use_vaw and os.path.isdir(args.vaw_dir):
        syn_lookup = auto_build_vaw_to_lvis_synonyms(
            args.vaw_dir, name_values_str, lvis_lookup, syn_lookup)
        print(f"  扩充后同义词: {len(syn_lookup)}")

    # ── STEP 2: COCO 路径 ──
    coco_roots = []
    if args.use_paco:
        input_roots = list(args.coco_images) if isinstance(args.coco_images, (list, tuple)) else [args.coco_images]
        for r in input_roots:
            if not os.path.isdir(r):
                print(f"  ⚠️ COCO 路径不存在: {r}");
                continue
            coco_roots.append(r)
            for sub in ("train2014", "val2014", "train2017", "val2017"):
                sp = os.path.join(r, sub)
                if os.path.isdir(sp) and sp not in coco_roots:
                    coco_roots.append(sp)
        if not coco_roots:
            print(f"  ❌ PACO 自动禁用 (无 COCO)");
            args.use_paco = False
        else:
            print(f"  COCO 路径展开: {len(coco_roots)} 个")

    # ── STEP 3: 收集 ──
    paco_triples = {};
    paco_obj_total = {};
    img_id_to_file = {}
    vaw_triples = {};
    vaw_obj_total = {};
    vaw_match_stats = Counter()
    vg_triples = {};
    vg_obj_total = {};
    vg_image_meta = {};
    vg_match_stats = Counter()

    if args.use_paco:
        paco_triples, paco_obj_total, img_id_to_file, _ = collect_paco_triples(
            args.paco_json, name_values_str, lvis_lookup, syn_lookup,
            dim_whitelist=active_dim_whitelist,
            caption_freq=caption_freq,
            freq_threshold=args.caption_freq_threshold,
        )
    if args.use_vaw:
        vaw_triples, vaw_obj_total, vaw_match_stats = collect_vaw_triples(
            args.vaw_dir, name_values_str, lvis_lookup, syn_lookup,
            unified_attr2dim,
            debug=args.debug_match)
    if args.use_vg:
        vg_triples, vg_obj_total, vg_image_meta, vg_match_stats = collect_vg_triples(
            args.vg_attributes_json, args.vg_image_data_json,
            name_values_str, lvis_lookup, syn_lookup,
            vg_attr2dim,
            max_images_per_attr=args.vg_max_per_attr,
            min_box_per_attr=args.vg_min_box_per_attr,
            debug=args.debug_match)

    if not paco_triples and not vaw_triples and not vg_triples:
        print(f"\n  ❌ 三源都没收集到, 退出");
        return

    # ── STEP 4: 合并 ──
    print(f"\n{'-' * 50}\nSTEP 4 | 合并 (加 min_dims_per_obj)\n{'-' * 50}")
    print(f"  ★ MAX_PER_DIM={args.max_per_dim}")
    print(f"  ★ MIN_DIMS_PER_OBJ={args.min_dims_per_obj}")

    paco_obj2dim2attrs = defaultdict(lambda: defaultdict(list))
    for (obj, dim, av), boxes in paco_triples.items():
        if len(boxes) >= args.min_images:
            paco_obj2dim2attrs[obj][dim].append((av, len(boxes)))
    paco_finals = select_attrs_adaptive(
        paco_obj2dim2attrs, paco_obj_total,
        min_ratio=args.min_ratio, min_count=args.min_count,
        max_per_dim=args.max_per_dim, max_per_obj=args.topk,
        min_dims_per_obj=args.min_dims_per_obj,
        source_tag="PACO")

    vaw_obj2dim2attrs = defaultdict(lambda: defaultdict(list))
    for (obj, dim, av), boxes in vaw_triples.items():
        if len(boxes) >= args.min_images:
            vaw_obj2dim2attrs[obj][dim].append((av, len(boxes)))
    vaw_finals = select_attrs_adaptive(
        vaw_obj2dim2attrs, vaw_obj_total,
        min_ratio=args.min_ratio, min_count=args.min_count,
        max_per_dim=args.max_per_dim, max_per_obj=args.topk,
        min_dims_per_obj=args.min_dims_per_obj,
        source_tag="VAW")

    vg_obj2dim2attrs = defaultdict(lambda: defaultdict(list))
    for (obj, dim, av), boxes in vg_triples.items():
        if len(boxes) >= args.min_images:
            vg_obj2dim2attrs[obj][dim].append((av, len(boxes)))
    vg_finals = select_attrs_adaptive(
        vg_obj2dim2attrs, vg_obj_total,
        min_ratio=args.min_ratio, min_count=args.min_count,
        max_per_dim=args.max_per_dim, max_per_obj=args.topk,
        min_dims_per_obj=args.min_dims_per_obj,
        source_tag="VG")

    print(f"\n  分源筛选: PACO={len(paco_finals)} VAW={len(vaw_finals)} VG={len(vg_finals)}")

    merged_triples = {};
    merged_source = {}
    final_triples = [];
    seen_keys = set()
    for (obj, dim, av, n) in paco_finals:
        key = (obj, dim, av)
        if key in seen_keys: continue
        seen_keys.add(key)
        merged_triples[key] = paco_triples[key]
        merged_source[key] = "PACO"
        final_triples.append((obj, dim, av, n))
    n_paco_kept = len(final_triples)
    n_vaw_added = 0
    for (obj, dim, av, n) in vaw_finals:
        key = (obj, dim, av)
        if key in seen_keys: continue
        seen_keys.add(key)
        merged_triples[key] = vaw_triples[key]
        merged_source[key] = "VAW"
        final_triples.append((obj, dim, av, n))
        n_vaw_added += 1
    n_vg_added = 0
    for (obj, dim, av, n) in vg_finals:
        key = (obj, dim, av)
        if key in seen_keys: continue
        seen_keys.add(key)
        merged_triples[key] = vg_triples[key]
        merged_source[key] = "VG"
        final_triples.append((obj, dim, av, n))
        n_vg_added += 1
    print(f"  合并后: PACO={n_paco_kept} VAW+={n_vaw_added} VG+={n_vg_added} 总={len(final_triples)}")

    valid = merged_triples
    n_attrs = len(final_triples)
    obj_attr_counts = Counter(t[0] for t in final_triples)
    dim_stats = Counter(t[1] for t in final_triples)

    print(f"  覆盖物体: {len(obj_attr_counts)}")
    if obj_attr_counts:
        cs = sorted(obj_attr_counts.values())
        print(f"  单物体属性数: min={cs[0]} p50={cs[len(cs) // 2]} max={cs[-1]} avg={n_attrs / len(cs):.1f}")
    print(f"\n  ★ dim 分布 (前 30, 共 {len(dim_stats)} 个):")
    for d, cnt in dim_stats.most_common(30):
        bar = '#' * min(30, cnt // max(1, n_attrs // 200))
        print(f"    {d:30s}: {cnt:>5}  {bar}")

    src_stats = Counter()
    for (obj, dim, av, _) in final_triples:
        src_stats[merged_source.get((obj, dim, av), "?")] += 1
    print(f"\n  来源: " + ", ".join(f"{s}={c}" for s, c in src_stats.most_common()))

    covered_lvis = set(obj_attr_counts.keys())
    cov_rate = len(covered_lvis) / max(len(unique_lvis_names), 1)
    print(f"  LVIS 覆盖率: {len(covered_lvis)}/{len(unique_lvis_names)} ({cov_rate:.1%})")

    # ── STEP 6: 编码器 ──
    print(f"\n{'-' * 50}\nSTEP 6 | 加载编码器\n{'-' * 50}")
    encoder = BLIP2Encoder(args.eva_clip, args.blip2_ckpt, DEVICE)
    if encoder.feature_dim != name_dim:
        print(f"  ❌ 维度不匹配: {encoder.feature_dim} vs {name_dim}");
        return

    # ── STEP 7: VG 路径 ──
    vg_root = None
    needs_vg = (args.use_vaw and any(merged_source.get((o, d, a)) == "VAW"
                                     for (o, d, a, _) in final_triples)) or \
               (args.use_vg and any(merged_source.get((o, d, a)) == "VG"
                                    for (o, d, a, _) in final_triples))
    if needs_vg:
        print(f"\n{'-' * 50}\nSTEP 7 | 探测 VG 路径\n{'-' * 50}")
        test_ids = []
        for (o, d, a, _) in final_triples:
            if merged_source.get((o, d, a)) in ("VAW", "VG"):
                test_ids = [iid for iid, _ in valid[(o, d, a)][:20]]
                break
        candidate_roots = [
            args.vg_images, os.path.join(args.vg_images, "VG"),
            os.path.join(os.path.dirname(args.vg_images), "VG"),
            "/root/autodl-tmp/wh5319/all_data/VG",
            "/root/autodl-tmp/wh5319/all_data/data/VG",
        ]
        seen = set()
        candidate_roots = [r for r in candidate_roots if r not in seen and not seen.add(r)]
        vg_root, found = probe_vg_root(test_ids[:20], candidate_roots)
        if found == 0:
            print(f"  ❌ 找不到 VG 图像");
            vg_root = None
        else:
            print(f"  ✅ VG: {vg_root} ({found}/20)")

    # ── STEP 8: 提取原型 ──
    print(f"\n{'-' * 50}\nSTEP 8 | 提取视觉原型\n{'-' * 50}")
    paco_finals_full = [t for t in final_triples if merged_source.get((t[0], t[1], t[2])) == "PACO"]
    vaw_finals_full = [t for t in final_triples if merged_source.get((t[0], t[1], t[2])) == "VAW"]
    vg_finals_full = [t for t in final_triples if merged_source.get((t[0], t[1], t[2])) == "VG"]

    all_attr_vectors = [];
    all_meta = []

    if paco_finals_full:
        print(f"\n  ── PACO 原型 ({len(paco_finals_full)}) ──")

        def paco_finder(image_id):
            return find_coco_image(image_id, coco_roots, img_id_to_file.get(image_id))

        vecs, metas = extract_prototypes(
            paco_finals_full, paco_triples, encoder, paco_finder,
            max_per_attr=args.max_per_attr, margin=args.margin, batch=args.batch,
            source_tag="PACO")
        all_attr_vectors.extend(vecs);
        all_meta.extend(metas)

    if vaw_finals_full and vg_root:
        print(f"\n  ── VAW 原型 ({len(vaw_finals_full)}) ──")

        def vaw_finder(image_id):
            return find_vg_image(image_id, vg_root)

        vecs, metas = extract_prototypes(
            vaw_finals_full, vaw_triples, encoder, vaw_finder,
            max_per_attr=args.max_per_attr, margin=args.margin, batch=args.batch,
            source_tag="VAW")
        all_attr_vectors.extend(vecs);
        all_meta.extend(metas)

    if vg_finals_full and vg_root:
        print(f"\n  ── VG 原型 ({len(vg_finals_full)}) ──")
        coco_roots_for_vg = coco_roots if args.use_paco else []

        def vg_finder(image_id):
            p = find_vg_image(image_id, vg_root)
            if p: return p
            meta = vg_image_meta.get(image_id)
            if meta and meta.get('coco_id') and coco_roots_for_vg:
                return find_coco_image(meta['coco_id'], coco_roots_for_vg)
            return None

        vecs, metas = extract_prototypes(
            vg_finals_full, vg_triples, encoder, vg_finder,
            max_per_attr=args.max_per_attr, margin=args.margin, batch=args.batch,
            source_tag="VG")
        all_attr_vectors.extend(vecs);
        all_meta.extend(metas)

    n_final = len(all_meta)
    if n_final == 0:
        print(f"  ❌ 没提取到任何原型");
        return
    attr_matrix = np.stack(all_attr_vectors).astype(np.float32)
    print(f"\n  最终 attr_matrix: {attr_matrix.shape}")
    del all_attr_vectors;
    gc.collect()

    # ── STEP 9: 桥接 + 保存 ──
    print(f"\n{'-' * 50}\nSTEP 9 | 桥接 + 保存\n{'-' * 50}")
    name2attr_canonical = defaultdict(list)
    attr2name_pkl_idx = {}
    for ai, m in enumerate(all_meta):
        name2attr_canonical[m['obj_name']].append(ai)
        pkl_idx = lvis_lookup.get(m['obj_name'], -1)
        attr2name_pkl_idx[ai] = [pkl_idx] if pkl_idx >= 0 else []
    name2attr_evcap = defaultdict(list)
    for obj, idxs in name2attr_canonical.items():
        key = _evcap_pre_name(obj)
        if key:
            name2attr_evcap[key].extend(idxs)
    name2attr_evcap = {k: sorted(set(v)) for k, v in name2attr_evcap.items()}
    obj_names_evcap = [_evcap_pre_name(m['obj_name']) for m in all_meta]
    obj_attr_map_evcap = {}
    for m in all_meta:
        key = _evcap_pre_name(m['obj_name'])
        obj_attr_map_evcap.setdefault(key, {}).setdefault(m['dim'], []).append(m['attr_value'])

    actual_dims = sorted(set(m['dim'] for m in all_meta))
    print(f"  v6.9.2 实际入库 dim 数: {len(actual_dims)}")
    print(f"    {actual_dims}")

    attr_pkl = {
        "keys": attr_matrix,
        "values": [m['text'] for m in all_meta],
        "attr_values": [m['attr_value'] for m in all_meta],
        "dims": [m['dim'] for m in all_meta],
        "obj_names": obj_names_evcap,
        "obj_names_canonical": [m['obj_name'] for m in all_meta],
        "n_images": [m['n_images'] for m in all_meta],
        "n_total": [m['n_total'] for m in all_meta],
        "sources": [m['source'] for m in all_meta],
        "name2attr_idx": dict(name2attr_evcap),
        "name2attr_idx_canonical": dict(name2attr_canonical),
        "attr2name_pkl_idx": {str(k): v for k, v in attr2name_pkl_idx.items()},
        "obj_attr_map": obj_attr_map_evcap,
        "dim": name_dim,
        "n_vectors": n_final,
        "n_objects": len(name2attr_evcap),
        "topk": args.topk, "min_images": args.min_images,
        "min_ratio": args.min_ratio, "min_count": args.min_count,
        "max_per_dim": args.max_per_dim,
        "encoding": ("bbox → LAVIS EVA-ViT-g [257,1408]"
                     " → LAVISQFormerForRetrieval [32,768] → mean → [768]"),
        "encoding_branch": "visual",
        "space_aligned": True,
        "version": "v6.9.2-aligned-with-classify-v3",
        "visual_encoder": "LAVIS eva_vit_g.pth",
        "qformer": "caption_model.retrieval_qformer.LAVISQFormerForRetrieval",
        "preprocess": "Resize((224,224)) BICUBIC + CLIP mean/std",
        "name_key_style": "evcap _evcap_pre_name",
        "data_sources": {
            "paco": args.use_paco, "vaw": args.use_vaw, "vg": args.use_vg,
            "n_paco_attrs": src_stats.get("PACO", 0),
            "n_vaw_attrs": src_stats.get("VAW", 0),
            "n_vg_attrs": src_stats.get("VG", 0),
        },
        "all_dims_in_lib": actual_dims,
        "n_dims": len(actual_dims),
        "n_unified_attr_table": len(unified_attr2dim),
        "n_vg_strict_attr_table": len(vg_attr2dim),
        # ★ v6.9.2 metadata
        "v692_changes": [
            f"v6.9.2: 适配 classify_vg_attrs_v3 输出, 修 v6.9.1 命名错配 bug",
            f"白名单大小: {len(CAPTION_FRIENDLY_DIMS)} (v6.9.1 是 43)",
            f"v6.9.1 bug: 'face_expression'/'hair_color'/'hair_length' (下划线) → 修为 VAW 原生空格",
            f"收纳 v3 新 dim: temperature/taste/wetness/lighting/season/emotion 等",
            f"实际入库: {len(actual_dims)} dim, {n_final:,} 个 attr 原型",
        ],
        "v691_changes": [
            f"双层过滤策略: PACO/VAW 普通 + VG 严格",
            f"PACO/VAW caption 词频门槛: {args.caption_freq_threshold}",
            f"VG 专用词频门槛:           {args.vg_caption_freq_threshold} (更严)",
            f"VG dim 限制在 VAW 集合内:  {not args.vg_no_restrict_to_vaw_dims}",
            f"VG box 范围:               [{args.vg_min_box_per_attr}, {args.vg_max_per_attr}]",
            f"PACO/VAW unified 表大小:   {len(unified_attr2dim):,} attr 词",
            f"VG 严格表大小:             {len(vg_attr2dim):,} attr 词",
        ],
        "v692_caption_friendly_dims": sorted(CAPTION_FRIENDLY_DIMS),
        "v692_caption_freq_threshold": args.caption_freq_threshold,
        "v692_min_dims_per_obj": args.min_dims_per_obj,
        "v691_vg_caption_freq_threshold": args.vg_caption_freq_threshold,
        "v691_vg_min_box_per_attr": args.vg_min_box_per_attr,
        "v691_vg_max_box_per_attr": args.vg_max_per_attr,
        "v691_vg_restrict_dims_to_vaw": (not args.vg_no_restrict_to_vaw_dims),
    }

    pkl_path = os.path.join(args.output, "attr_memory.pkl")
    with open(pkl_path, 'wb') as f:
        pickle.dump(attr_pkl, f, protocol=4)
    print(f"  attr_memory.pkl: {os.path.getsize(pkl_path) / 1e6:.1f} MB, shape {attr_matrix.shape}")

    json_path = os.path.join(args.output, "name2attr_topk.json")
    topk_json = {}
    for obj_key in sorted(obj_attr_map_evcap.keys()):
        canon_name = next((m['obj_name'] for m in all_meta
                           if _evcap_pre_name(m['obj_name']) == obj_key), obj_key)
        topk_json[obj_key] = {
            "n_attrs": sum(len(v) for v in obj_attr_map_evcap[obj_key].values()),
            "attr_indices": name2attr_evcap[obj_key],
            "name_pkl_index": lvis_lookup.get(canon_name, -1),
            "dims": obj_attr_map_evcap[obj_key],
        }
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(topk_json, f, indent=2, ensure_ascii=False)
    print(f"  name2attr_topk.json: {os.path.getsize(json_path) / 1e3:.0f} KB")

    print(f"\n{SEP}\n  v6.9.2 构建完成 ✅\n{SEP}")
    print(f"  attr 总向量:    {n_final:,}")
    print(f"  覆盖物体:       {len(name2attr_evcap)}")
    print(f"  ★ dim 类别数:   {len(actual_dims)}  (v6.9.2 适配 v3 classify)")
    print(f"  ★ 白名单:       {len(CAPTION_FRIENDLY_DIMS)} 个 dim")
    print(f"  ★ MAX_PER_DIM:  {args.max_per_dim}")
    print(f"  ★ MIN_DIMS_PER_OBJ: {args.min_dims_per_obj}")
    print(f"  ★ PACO/VAW freq:    >= {args.caption_freq_threshold}")
    print(f"  ★ VG freq:          >= {args.vg_caption_freq_threshold}")
    print(f"  ★ VG box 范围:      [{args.vg_min_box_per_attr}, {args.vg_max_per_attr}]")
    print(f"  ★ VG dim ⊂ VAW:    {not args.vg_no_restrict_to_vaw_dims}")
    print(f"  LVIS 覆盖率:    {cov_rate:.1%}")
    print(f"  来源: " + ", ".join(f"{s}={c}({c / n_attrs * 100:.1f}%)" for s, c in src_stats.most_common()))
    print(f"{SEP}")


if __name__ == "__main__":
    main()