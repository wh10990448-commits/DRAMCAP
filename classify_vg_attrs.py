#!/usr/bin/env python3
"""
classify_vg_attrs_v3.py  —  VG attr 分类 (v3: 多层噪声过滤)

═════════════════════════════════════════════════════════════════════════
v3 关键改动 (相对 v2):

1. ★ COCO caption 词频过滤 (核心, 跟 build_attr_memory v6.9 同款)
   - 读 captions_train2014.json, 统计每个词的频次
   - VG attr 词的所有组成单词必须在 caption 出现 >= N 次 (默认 30)
   - 砍掉 caption 里几乎不出现的长尾噪声词

2. ★ 多层启发式过滤 (新增 _filter_advanced_noise)
   - 包含数字: '12 inch', '5 feet' → 丢
   - 特殊字符: '@', '#', URL → 丢
   - 序数词:   'first', '1st', '2nd' → 丢
   - 专有名词: 'nike', 'adidas', 'monday' → 丢
   - 超长短语: ≥ 5 词的短语 → 丢

3. ★ 黑名单加强
   - 加序数词列表 _ORDINAL_BLACKLIST
   - 加专有名词列表 _PROPER_NOUN_BLACKLIST (品牌/国家/星期/月份)

4. ★ per-dim 数量上限 (默认 200)
   - 防止某个 dim 收集到几千个 attr
   - 按 caption 词频排序保留 top-N

5. ★ --no_misc 选项
   - 默认 misc dim 也输出 (跟 v2 一致)
   - --no_misc 启用时, misc 整个 dim 丢弃

v2 继承不变:
- VAW 优先去重
- SentenceBERT 分类
- _VG_NOISE_BLACKLIST 基础过滤
- DIM_ANCHORS 锚点

═════════════════════════════════════════════════════════════════════════
用法:

  # 推荐 (跟 build_attr_memory v6.9 配合):
  python classify_vg_attrs_v3.py \
      --vg_attributes_json /root/autodl-tmp/wh5319/all_data/VG/attributes.json \
      --vaw_attribute_types /root/autodl-tmp/wh5319/all_data/data/attribute_types.json \
      --coco_caption_json /root/autodl-tmp/wh5319/all_data/coco_2014/annotations/captions_train2014.json \
      --output_json /root/autodl-tmp/wh5319/all_data/VG/vg_attribute_types.json \
      --min_freq 50 \
      --caption_freq_threshold 30 \
      --sim_threshold 0.35 \
      --max_per_dim 200 \
      --no_misc

  # 离线 (本地 SentenceBERT):
  python classify_vg_attrs_v3.py \
      --vg_attributes_json ... \
      --vaw_attribute_types ... \
      --coco_caption_json ... \
      --output_json ... \
      --model_path /path/to/all-MiniLM-L6-v2/

  # 不用 caption 词频 (回退 v2 行为):
  python classify_vg_attrs_v3.py \
      --vg_attributes_json ... \
      --vaw_attribute_types ... \
      --output_json ... \
      --caption_freq_threshold 0
"""

import os
import re
import sys
import json
import argparse
from collections import Counter, defaultdict
from typing import Dict, List, Set, Optional


# ════════════════════════════════════════════════════════════════════
# 路径默认值
# ════════════════════════════════════════════════════════════════════
COCO_CAPTION_JSON = "/root/autodl-tmp/wh5319/all_data/coco_2014/annotations/captions_train2014.json"


# ════════════════════════════════════════════════════════════════════
# DIM 设计: VAW 42 + VG 新加 52 = ~94 dim 候选
# ════════════════════════════════════════════════════════════════════
DIM_ANCHORS: Dict[str, List[str]] = {
    # ════════════════ VAW 原 42 个 subtype (跟 VAW 完全对齐) ═══════════════
    # ── color 系 ──
    "color":           ["color", "red", "blue", "green", "yellow", "white", "black", "purple", "pink"],
    "hair color":      ["hair color", "blonde", "brunette", "redhead", "gray hair", "black hair"],
    "skin color":      ["skin color", "pale skin", "tan skin", "dark skin", "fair skin"],
    "wearing color":   ["clothing color", "outfit color", "dress color", "shirt color"],
    "letter color":    ["letter color", "text color", "font color", "writing color"],
    "tone":            ["tone", "shade", "tint", "hue", "color tone"],
    "color quantity":  ["multicolor", "monochrome", "two-toned", "colorful", "single color"],
    "brightness":      ["brightness", "bright", "dark", "dim", "luminous"],

    # ── size 系 ──
    "size":            ["size", "large", "small", "big", "tiny", "huge", "medium"],
    "height":          ["height", "tall", "short", "high", "low"],
    "length":          ["length", "long", "short length", "elongated"],
    "width":           ["width", "wide", "narrow", "broad", "thin width"],
    "fatness":         ["fatness", "fat", "skinny", "chubby", "slim", "thick body"],
    "thickness":       ["thickness", "thick", "thin", "slim", "dense"],
    "depth":           ["depth", "deep", "shallow", "profound"],
    "size comparison": ["bigger than", "smaller than", "larger", "smaller", "equal size"],
    "weight":          ["weight", "heavy", "light", "lightweight", "weighty"],
    "hair length":     ["hair length", "long hair", "short hair", "medium hair"],

    # ── 形状/纹理 ──
    "shape":           ["shape", "round", "square", "rectangular", "oval", "circular", "triangular"],
    "orientation":     ["orientation", "vertical", "horizontal", "tilted", "upright"],
    "pattern":         ["pattern", "striped", "spotted", "checkered", "plaid", "floral"],
    "texture":         ["texture", "smooth", "rough", "bumpy", "soft", "hard"],
    "hardness":        ["hardness", "hard", "soft", "firm", "flexible", "rigid"],
    "opaqeness":       ["opacity", "transparent", "translucent", "opaque", "clear"],
    "material":        ["material", "wooden", "metal", "plastic", "glass", "leather", "fabric"],

    # ── 状态 ──
    "cleanliness":     ["cleanliness", "clean", "dirty", "spotless", "filthy", "tidy"],
    "maturity":        ["maturity", "mature", "young", "adult", "elderly", "juvenile"],
    "newness":         ["newness", "new", "old", "brand new", "ancient", "antique"],
    "state":           ["state", "broken", "intact", "damaged", "complete"],
    "weather":         ["weather", "sunny", "rainy", "cloudy", "snowy", "windy", "foggy"],
    "machine state":   ["machine running", "powered on", "powered off", "operating", "idle"],

    # ── 动作 ──
    "hand movement":   ["hand movement", "waving", "pointing", "grabbing", "holding"],
    "pose":            ["pose", "sitting", "standing", "lying", "kneeling", "crouching"],
    "activity":        ["activity", "playing", "working", "reading", "writing", "eating"],
    "sport activity":  ["sport activity", "running", "swimming", "skiing", "skating", "jumping"],
    "face pose":       ["face pose", "facing", "looking up", "looking down", "side view"],
    "face expression": ["face expression", "smiling", "frowning", "laughing", "crying", "neutral"],

    # ── 位置/属性 ──
    "location":        ["location", "indoor", "outdoor", "inside", "outside"],
    "place":           ["place", "kitchen", "bedroom", "park", "beach", "street"],
    "wearing accessories": ["accessories", "wearing hat", "wearing glasses", "wearing jewelry"],
    "closeness":       ["closeness", "close", "far", "near", "distant", "adjacent"],
    "gender":          ["gender", "male", "female", "masculine", "feminine"],
    "race":            ["race", "asian", "african", "caucasian", "hispanic"],

    # ════════════════ VG 新加 52 个 dim (VAW 没覆盖的) ═══════════════
    # ── 物理感知 ──
    "temperature":     ["temperature", "hot", "cold", "warm", "cool", "freezing", "boiling", "icy"],
    "taste":           ["taste", "sweet", "sour", "salty", "bitter", "spicy", "savory", "tasty"],
    "smell":           ["smell", "fragrant", "stinky", "smelly", "scented", "musty", "fresh smell"],
    "sound":           ["sound", "loud", "quiet", "silent", "noisy", "noiseless"],
    "speed":           ["speed", "fast", "slow", "rapid", "swift", "speedy", "sluggish"],

    # ── 状态细分 ──
    "wetness":         ["wetness", "wet", "dry", "damp", "soaked", "moist", "drenched"],
    "fullness":        ["fullness", "full", "empty", "half-full", "stuffed", "vacant"],
    "freshness":       ["freshness", "fresh", "stale", "rotten", "spoiled", "moldy"],
    "ripeness":        ["ripeness", "ripe", "unripe", "overripe", "green fruit", "mature fruit"],
    "openness":        ["openness", "open", "closed", "shut", "ajar", "sealed"],
    "lit_state":       ["lit", "unlit", "lighted", "glowing", "off", "on", "burning"],
    "operational":     ["operational", "working", "broken", "functional", "non-functional"],
    "completion":      ["completion", "complete", "incomplete", "partial", "whole", "finished"],

    # ── 视觉细分 ──
    "saturation":      ["saturation", "vibrant", "dull", "vivid", "muted", "intense color"],
    "finish":          ["finish", "glossy", "matte", "satin", "polished surface"],
    "reflectivity":    ["reflectivity", "reflective", "mirror-like", "shiny", "non-reflective"],
    "luminosity":      ["luminosity", "glowing", "luminous", "radiant", "shining"],
    "transparency_level": ["transparency", "see-through", "frosted", "tinted", "crystal clear"],

    # ── 几何细分 ──
    "edge_type":       ["edge", "sharp", "blunt", "jagged", "rounded edge", "pointed edge"],
    "symmetry":        ["symmetry", "symmetric", "asymmetric", "balanced", "uneven"],
    "angle":           ["angle", "angular", "sharp angle", "right angle", "obtuse"],
    "curvature":       ["curvature", "curved", "straight", "bent", "twisted"],
    "profile":         ["profile", "side profile", "front profile", "outline shape"],

    # ── 排列/数量 ──
    "stack":           ["stacked", "piled", "scattered", "arranged", "organized", "messy"],
    "density":         ["density", "dense", "sparse", "concentrated", "diluted", "thin"],
    "distribution":    ["distribution", "scattered", "clustered", "spread out", "grouped"],
    "repetition":      ["repetition", "repeated", "uniform", "random", "patterned"],
    "alignment":       ["alignment", "aligned", "misaligned", "parallel", "perpendicular"],

    # ── 食物特定 ──
    "doneness":        ["doneness", "raw", "cooked", "rare", "well-done", "burnt"],
    "cooking_state":   ["cooking state", "fried", "baked", "boiled", "grilled", "roasted"],
    "preparation":     ["preparation", "sliced", "chopped", "whole", "diced", "shredded"],

    # ── 人物细分 ──
    "emotion":         ["emotion", "happy", "sad", "angry", "joyful", "depressed"],
    "hairstyle":       ["hairstyle", "curly", "straight hair", "wavy", "braided", "bald"],
    "body_position":   ["body position", "leaning", "bending", "stretching", "resting"],
    "hand_gesture":    ["hand gesture", "thumbs up", "pointing", "waving", "fist"],
    "facial_feature":  ["facial feature", "bearded", "mustache", "freckled", "dimpled"],
    "eye_state":       ["eye state", "open eyes", "closed eyes", "wide eyes", "squinting"],
    "mouth_state":     ["mouth state", "open mouth", "closed mouth", "smiling mouth"],

    # ── 车辆/物品 ──
    "motion_state":    ["motion state", "moving", "stationary", "parked", "stopped"],
    "door_state":      ["door state", "door open", "door closed", "ajar door", "locked door"],
    "light_state":     ["light state", "lights on", "lights off", "flashing", "headlights on"],

    # ── 环境 ──
    "lighting":        ["lighting", "well-lit", "dim lighting", "natural light", "artificial light"],
    "time_of_day":     ["time of day", "morning", "afternoon", "evening", "night", "dusk", "dawn"],
    "season":          ["season", "spring", "summer", "autumn", "winter", "fall"],

    # ── 价值/质量 ──
    "value_judgment":  ["value", "expensive", "cheap", "valuable", "premium", "luxury"],
    "rarity":          ["rarity", "rare", "common", "unique", "ordinary", "special"],
    "quality":         ["quality", "high quality", "low quality", "premium", "inferior"],
    "formality":       ["formality", "formal", "casual", "elegant", "informal"],

    # ── 功能 ──
    "functionality":   ["functionality", "functional", "non-functional", "useful", "decorative"],
    "usage":           ["usage", "in use", "unused", "active", "idle"],
    "accessibility":   ["accessibility", "accessible", "blocked", "reachable", "hidden"],
}


# ════════════════════════════════════════════════════════════════════
# VG 噪声黑名单 (基础, 跟 v2 一致)
# ════════════════════════════════════════════════════════════════════
_VG_NOISE_BLACKLIST = {
    "a", "an", "the", "this", "that", "these", "those",
    "it", "its", "his", "her", "their", "him", "she", "he", "they",
    "very", "too", "really", "quite", "so", "also", "just", "even",
    "more", "most", "less", "least",
    "of", "in", "on", "at", "by", "to", "for", "with", "from", "about",
    "and", "or", "but", "if", "as", "than",
    "is", "are", "was", "were", "be", "been", "being",
    "has", "have", "had", "having", "do", "does", "did", "doing",
    "can", "could", "will", "would", "should", "may", "might",
    "many", "few", "some", "several", "any", "all", "each", "every",
    "another", "both", "either", "neither", "lot", "lots",
    "one", "two", "three", "four", "five", "six", "seven", "eight",
    "nine", "ten",
    "no", "not", "yes",
    "what", "where", "when", "why", "how", "who", "which",
    "there", "here", "now", "then", "ever", "never", "always", "sometimes",
    "nice", "beautiful", "ugly", "good", "bad", "cool", "cute", "pretty",
    "normal", "regular", "typical", "common", "other", "different", "same",
    "similar", "interesting", "boring", "unusual",
    "in the air", "on the wall", "on the ground", "at the top", "at the bottom",
    "in the background", "in the foreground", "at table",
    "out of", "next to", "in front", "in front of", "behind it",
    "is on", "is in", "is at", "is up", "is down",
}


# ════════════════════════════════════════════════════════════════════
# v3 新增: 序数词黑名单
# ════════════════════════════════════════════════════════════════════
_ORDINAL_BLACKLIST = {
    "first", "second", "third", "fourth", "fifth",
    "sixth", "seventh", "eighth", "ninth", "tenth",
    "1st", "2nd", "3rd", "4th", "5th",
    "6th", "7th", "8th", "9th", "10th", "11th", "12th",
    "next", "previous", "last", "final", "beginning", "ending",
    "former", "latter",
}


# ════════════════════════════════════════════════════════════════════
# v3 新增: 专有名词黑名单 (品牌/国家/星期/月份/宗教)
# ════════════════════════════════════════════════════════════════════
_PROPER_NOUN_BLACKLIST = {
    # ── 品牌 ──
    "nike", "adidas", "puma", "reebok", "newbalance", "under armour",
    "sony", "samsung", "apple", "lg", "panasonic", "nintendo", "microsoft",
    "toyota", "honda", "ford", "bmw", "mercedes", "audi", "lexus", "kia",
    "chevy", "chevrolet", "volvo", "tesla", "porsche", "ferrari",
    "coca cola", "coca-cola", "pepsi", "sprite", "fanta",
    "starbucks", "mcdonald", "mcdonalds", "kfc", "subway", "burger king",
    "google", "facebook", "twitter", "instagram",
    "ikea", "walmart", "target", "amazon",

    # ── 国家/地区 (作为 attr 时多是噪声) ──
    "american", "americans",
    "chinese", "japanese", "korean", "thai", "vietnamese",
    "italian", "italians", "french", "german", "spanish", "russian",
    "indian", "indians", "pakistani",
    "english", "british", "scottish", "irish",
    "asian", "asians", "african", "africans", "european", "europeans",
    "mexican", "mexicans", "brazilian", "argentinian",
    "australian", "canadian", "egyptian", "greek", "turkish",

    # ── 星期 ──
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
    "weekday", "weekend",

    # ── 月份 ──
    "january", "february", "march", "april", "may", "june", "july",
    "august", "september", "october", "november", "december",

    # ── 宗教 ──
    "christian", "jewish", "muslim", "buddhist", "hindu", "catholic",

    # ── 节日 ──
    "christmas", "halloween", "thanksgiving", "easter", "valentine",

    # ── 政党/组织 (caption 几乎不写) ──
    "democratic", "republican", "police", "military",
}


def _normalize(s: str) -> str:
    return s.lower().strip().replace("_", " ").replace("-", " ")


# ════════════════════════════════════════════════════════════════════
# v3 新增: COCO caption 词频统计
# ════════════════════════════════════════════════════════════════════
def build_caption_word_freq(coco_caption_json: str) -> Optional[Counter]:
    """统计 COCO caption 词频, 用于 attr 词过滤"""
    if not os.path.exists(coco_caption_json):
        print(f"  ⚠️ COCO caption json 不存在: {coco_caption_json}")
        print(f"     v3 caption 词频过滤将被禁用")
        return None
    print(f"  [v3 caption-freq] 统计 COCO caption 词频 ...")
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
    print(f"    前 20 高频词: {[w for w,_ in counter.most_common(20)]}")
    return counter


def filter_by_caption_freq(vg_attrs: Counter, caption_freq: Optional[Counter],
                            min_freq: int) -> Counter:
    """v3: 用 caption 词频过滤 attr 候选"""
    if caption_freq is None or min_freq <= 0:
        print(f"  [caption 词频过滤] 已禁用 (caption_freq={caption_freq is not None}, min_freq={min_freq})")
        return vg_attrs
    filtered = Counter()
    n_dropped = 0
    dropped_examples = []
    for av, c in vg_attrs.items():
        words = av.split()
        if all(caption_freq.get(w, 0) >= min_freq for w in words):
            filtered[av] = c
        else:
            n_dropped += 1
            if len(dropped_examples) < 10:
                dropped_examples.append(av)
    print(f"  [caption 词频过滤 (>= {min_freq})]")
    print(f"    {len(vg_attrs):,} → {len(filtered):,} (-{n_dropped:,})")
    if dropped_examples:
        print(f"    丢弃示例: {dropped_examples}")
    return filtered


# ════════════════════════════════════════════════════════════════════
# v3 新增: 多层启发式过滤
# ════════════════════════════════════════════════════════════════════
def _has_digit(s: str) -> bool:
    return any(c.isdigit() for c in s)


def _has_special_char(s: str) -> bool:
    return any(c in s for c in "@#$%^&*<>{}[]|\\/`~+=")


def _is_too_long(s: str, max_words: int = 4) -> bool:
    """超过 N 个词的短语大概率是描述句不是 attr"""
    return len(s.split()) > max_words


def filter_advanced_noise(vg_attrs: Counter, max_words: int = 4) -> Counter:
    """v3: 多层启发式过滤 (数字/特殊字符/序数词/专有名词/超长短语)"""
    filtered = Counter()
    stats = Counter()
    examples_per_kind: Dict[str, List[str]] = defaultdict(list)

    for av, c in vg_attrs.items():
        # 1) 包含数字 ('12 inch', '5 feet', '3d')
        if _has_digit(av):
            stats['digit'] += 1
            if len(examples_per_kind['digit']) < 5:
                examples_per_kind['digit'].append(av)
            continue
        # 2) 特殊字符 ('@user', '#tag', URL)
        if _has_special_char(av):
            stats['special_char'] += 1
            if len(examples_per_kind['special_char']) < 5:
                examples_per_kind['special_char'].append(av)
            continue
        # 3) 序数词
        if av in _ORDINAL_BLACKLIST:
            stats['ordinal'] += 1
            if len(examples_per_kind['ordinal']) < 5:
                examples_per_kind['ordinal'].append(av)
            continue
        # 4) 专有名词
        if av in _PROPER_NOUN_BLACKLIST:
            stats['proper_noun'] += 1
            if len(examples_per_kind['proper_noun']) < 5:
                examples_per_kind['proper_noun'].append(av)
            continue
        # 5) 超长短语
        if _is_too_long(av, max_words=max_words):
            stats['long_phrase'] += 1
            if len(examples_per_kind['long_phrase']) < 5:
                examples_per_kind['long_phrase'].append(av)
            continue
        filtered[av] = c

    n_dropped = len(vg_attrs) - len(filtered)
    print(f"  [v3 高级过滤]")
    print(f"    {len(vg_attrs):,} → {len(filtered):,} (-{n_dropped:,})")
    for kind, cnt in stats.most_common():
        examples = examples_per_kind[kind]
        ex_str = f" 示例: {examples}" if examples else ""
        print(f"    {kind:15s}: {cnt:,}{ex_str}")
    return filtered


# ════════════════════════════════════════════════════════════════════
# v2 不变: VAW 加载 + 去重
# ════════════════════════════════════════════════════════════════════
def load_vaw_attr_set(vaw_attribute_types_path: str) -> Set[str]:
    """读 VAW attribute_types.json, 返回所有 attr 词的 set (用于 VG 去重)"""
    if not os.path.exists(vaw_attribute_types_path):
        print(f"  ⚠️ VAW attribute_types.json 不存在: {vaw_attribute_types_path}")
        print(f"     不去重, VG 全部 attr 词都会进 vg_attribute_types.json")
        return set()
    with open(vaw_attribute_types_path) as f:
        vaw_types = json.load(f)
    attr_set = set()
    for vaw_dim, attrs in vaw_types.items():
        for a in attrs:
            attr_set.add(_normalize(a))
    print(f"  [VAW] 加载 {len(vaw_types)} 个 dim, {len(attr_set)} 个 attr 词 (用于 VG 去重)")
    return attr_set


def collect_vg_attr_freq(attributes_json: str, min_freq: int = 50) -> Counter:
    """读 VG attributes.json, 统计每个 attr 词出现频次, 返回过 min_freq 的"""
    print(f"  读 attributes.json (~440 MB)... ", flush=True)
    with open(attributes_json) as f:
        vg_data = json.load(f)
    print(f"  → {len(vg_data):,} 张图")

    print(f"  统计 attr 词频 ...", flush=True)
    cnt = Counter()
    for img_entry in vg_data:
        for obj in img_entry.get('attributes', []):
            for attr in (obj.get('attributes', []) or []):
                av = _normalize(attr)
                if not av:
                    continue
                cnt[av] += 1

    print(f"  总 attr 词去重数: {len(cnt):,}")
    print(f"  词频分布:")
    for thresh in [1000, 500, 100, 50, 20, 10]:
        n = sum(1 for c in cnt.values() if c >= thresh)
        print(f"    ≥ {thresh:>5}: {n:,}")

    filtered = Counter()
    n_blocked_freq = n_blocked_blk = n_blocked_short = 0
    for av, c in cnt.items():
        if c < min_freq:
            n_blocked_freq += 1
            continue
        if av in _VG_NOISE_BLACKLIST:
            n_blocked_blk += 1
            continue
        if len(av) < 2:
            n_blocked_short += 1
            continue
        filtered[av] = c

    print(f"\n  筛选后 (freq≥{min_freq}, 非黑名单, len≥2): {len(filtered):,} 个 attr 候选")
    print(f"    频次过滤掉: {n_blocked_freq:,}, 黑名单过滤掉: {n_blocked_blk:,}, 太短: {n_blocked_short:,}")
    return filtered


def deduplicate_against_vaw(vg_attrs: Counter, vaw_attr_set: Set[str]) -> Counter:
    """v2: 把 VAW 已有的 attr 词从 VG 候选里删掉 (VAW 优先)"""
    if not vaw_attr_set:
        print(f"  [去重] VAW attr 集合为空, 不去重")
        return vg_attrs
    deduped = Counter()
    n_removed = 0
    for av, c in vg_attrs.items():
        if av in vaw_attr_set:
            n_removed += 1
            continue
        deduped[av] = c
    print(f"  [去重] VG 候选 {len(vg_attrs):,} → 去掉 VAW 已有 {n_removed:,} → {len(deduped):,} 个待分类")
    return deduped


# ════════════════════════════════════════════════════════════════════
# SentenceBERT (跟 v2 一致)
# ════════════════════════════════════════════════════════════════════
def load_sentence_bert(model_path: str = None):
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("\n  ❌ 未安装 sentence-transformers")
        print("     pip install sentence-transformers --break-system-packages")
        sys.exit(1)

    name = model_path or 'sentence-transformers/all-MiniLM-L6-v2'
    print(f"\n  加载 SentenceBERT: {name}")
    if model_path and not os.path.exists(model_path):
        print(f"  ⚠️ 本地路径 {model_path} 不存在, 尝试在线下载")
        name = 'sentence-transformers/all-MiniLM-L6-v2'

    try:
        model = SentenceTransformer(name)
    except Exception as e:
        print(f"  ❌ 加载失败: {e}")
        print(f"\n  离线方案:")
        print(f"  1) 在能联网的机器上: git clone https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2")
        print(f"  2) 拷过来后: python classify_vg_attrs_v3.py --model_path /path/to/all-MiniLM-L6-v2/ ...")
        sys.exit(1)

    print(f"  ✅ 模型加载完成 (输出维度: {model.get_sentence_embedding_dimension()})")
    return model


def build_dim_centroids(model, anchors: Dict[str, List[str]]):
    import numpy as np
    print(f"\n  计算 {len(anchors)} 个 dim 的锚点 embedding ...")
    dim_names = list(anchors.keys())
    centroids = []
    for dn in dim_names:
        anchor_words = anchors[dn]
        embs = model.encode(anchor_words, show_progress_bar=False, normalize_embeddings=True)
        centroid = embs.mean(axis=0)
        centroid = centroid / (np.linalg.norm(centroid) + 1e-8)
        centroids.append(centroid)
    centroids = np.stack(centroids).astype(np.float32)
    print(f"  → centroid matrix: {centroids.shape}")
    return dim_names, centroids


def classify_attrs(model, dim_names, centroids, attr_words: List[str],
                   sim_threshold: float = 0.35, batch_size: int = 256):
    import numpy as np
    print(f"\n  分类 {len(attr_words):,} 个 attr 词 ...")
    attr_embs = model.encode(attr_words, batch_size=batch_size,
                             show_progress_bar=True, normalize_embeddings=True)
    attr_embs = attr_embs.astype(np.float32)
    sims = attr_embs @ centroids.T
    best_dim_idx = sims.argmax(axis=1)
    best_sim = sims.max(axis=1)

    dim_to_attrs = defaultdict(list)
    misc_count = 0
    sim_dist = []
    for i, av in enumerate(attr_words):
        s = float(best_sim[i])
        sim_dist.append(s)
        if s < sim_threshold:
            dim_to_attrs["misc"].append(av)
            misc_count += 1
        else:
            d = dim_names[best_dim_idx[i]]
            dim_to_attrs[d].append(av)

    sim_dist = np.array(sim_dist)
    print(f"\n  cos sim 分布: min={sim_dist.min():.3f} mean={sim_dist.mean():.3f} max={sim_dist.max():.3f}")
    print(f"  阈值 {sim_threshold}: {misc_count} 词进 misc, {len(attr_words)-misc_count} 词进有效 dim")
    return dim_to_attrs


# ════════════════════════════════════════════════════════════════════
# v3 新增: per-dim 数量上限 + misc 屏蔽
# ════════════════════════════════════════════════════════════════════
def cap_per_dim(dim_to_attrs: Dict[str, List[str]], freq: Counter,
                max_per_dim: int = 200) -> Dict[str, List[str]]:
    """v3: 每个 dim 的 attr 数量上限, 按词频排序保留 top-N"""
    capped = {}
    n_total_dropped = 0
    n_capped_dims = 0
    for d, attrs in dim_to_attrs.items():
        if len(attrs) > max_per_dim:
            sorted_attrs = sorted(attrs, key=lambda a: -freq.get(a, 0))
            capped[d] = sorted_attrs[:max_per_dim]
            n_total_dropped += len(attrs) - max_per_dim
            n_capped_dims += 1
        else:
            capped[d] = list(attrs)
    if n_total_dropped > 0:
        print(f"\n  [v3 per-dim cap] 上限 {max_per_dim}, "
              f"{n_capped_dims} 个 dim 被截断, 丢弃 {n_total_dropped:,} 个低频 attr")
    else:
        print(f"\n  [v3 per-dim cap] 上限 {max_per_dim}, 无 dim 超限, 不截断")
    return capped


def remove_misc_dim(dim_to_attrs: Dict[str, List[str]]) -> Dict[str, List[str]]:
    """v3: 丢弃 misc dim (sim < threshold 的兜底分类, 噪声大)"""
    if "misc" in dim_to_attrs:
        n_misc = len(dim_to_attrs["misc"])
        del dim_to_attrs["misc"]
        print(f"  [v3 --no_misc] 丢弃 misc dim 的 {n_misc:,} 个 attr 词")
    return dim_to_attrs


def print_summary(dim_to_attrs, freq, top_n: int = 30):
    print(f"\n  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f"  分类结果概览 (按 dim 内 attr 数排序)")
    print(f"  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    sorted_dims = sorted(dim_to_attrs.items(), key=lambda x: -len(x[1]))
    print(f"  共 {len(dim_to_attrs)} 个有效 dim, 总分类 {sum(len(v) for v in dim_to_attrs.values())} 个 attr 词\n")
    for dn, attrs in sorted_dims:
        sorted_attrs = sorted(attrs, key=lambda a: -freq.get(a, 0))
        sample = sorted_attrs[:8]
        sample_str = ", ".join(sample)
        suffix = f", ... (+{len(attrs)-8} more)" if len(attrs) > 8 else ""
        print(f"    {dn:25s} ({len(attrs):>3} 词): {sample_str}{suffix}")


def save_output(dim_to_attrs, freq, output_path: str):
    sorted_output = {}
    for dn in sorted(dim_to_attrs.keys()):
        attrs = dim_to_attrs[dn]
        attrs_sorted = sorted(attrs, key=lambda a: -freq.get(a, 0))
        sorted_output[dn] = attrs_sorted
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(sorted_output, f, indent=2, ensure_ascii=False)
    print(f"\n  ✅ 写入 {output_path}")
    print(f"     ({os.path.getsize(output_path)/1024:.1f} KB)")


def main():
    p = argparse.ArgumentParser(description="VG attr → VAW 格式 (v3: 多层噪声过滤)")
    p.add_argument("--vg_attributes_json", required=True)
    p.add_argument("--vaw_attribute_types", required=True,
                   help="VAW attribute_types.json (用于去重 VAW 已有的 attr 词)")
    p.add_argument("--output_json", required=True)
    p.add_argument("--min_freq", type=int, default=50,
                   help="VG 内 attr 词最低频次 (默认 50)")
    p.add_argument("--sim_threshold", type=float, default=0.35,
                   help="cos sim 阈值, 低于此值进 misc (默认 0.35)")
    p.add_argument("--model_path", default=None)
    p.add_argument("--batch_size", type=int, default=256)

    # ★ v3 新增参数
    p.add_argument("--coco_caption_json", default=COCO_CAPTION_JSON,
                   help="★ v3: COCO captions_train2014.json 路径")
    p.add_argument("--caption_freq_threshold", type=int, default=30,
                   help="★ v3: attr 词在 COCO caption 最低频次 (0=禁用, 默认 30)")
    p.add_argument("--max_per_dim", type=int, default=200,
                   help="★ v3: 每个 dim 最多保留多少 attr 词 (默认 200)")
    p.add_argument("--no_misc", action="store_true",
                   help="★ v3: 丢弃 misc dim (sim < threshold 的兜底分类)")
    p.add_argument("--max_words_per_attr", type=int, default=4,
                   help="★ v3: attr 短语最大词数 (默认 4, 超过丢)")

    args = p.parse_args()

    SEP = "=" * 70
    print(f"\n{SEP}")
    print(f"  VG attr → vg_attribute_types.json 分类器 (v3: 多层噪声过滤)")
    print(f"{SEP}")
    print(f"\n  [配置]")
    print(f"    VG attributes.json:       {args.vg_attributes_json}")
    print(f"    VAW attribute_types:      {args.vaw_attribute_types}")
    print(f"    COCO caption json:        {args.coco_caption_json}")
    print(f"    输出文件:                 {args.output_json}")
    print(f"    VG 内最小词频:            {args.min_freq}")
    print(f"    caption 词频门槛:         {args.caption_freq_threshold}  ★ v3")
    print(f"    cos sim 阈值:             {args.sim_threshold}")
    print(f"    总 dim 候选数:            {len(DIM_ANCHORS)}")
    print(f"    每 dim 上限:              {args.max_per_dim}  ★ v3")
    print(f"    短语最大词数:             {args.max_words_per_attr}  ★ v3")
    print(f"    丢弃 misc dim:            {args.no_misc}  ★ v3")

    # ── STEP 1: 加载 VAW attr set (去重用) ──
    print(f"\n{'-'*60}\nSTEP 1 | 加载 VAW attr 词集合 (用于去重)\n{'-'*60}")
    vaw_attr_set = load_vaw_attr_set(args.vaw_attribute_types)

    # ── STEP 2: 统计 VG 词频 ──
    print(f"\n{'-'*60}\nSTEP 2 | 统计 VG attr 词频\n{'-'*60}")
    freq_raw = collect_vg_attr_freq(args.vg_attributes_json, min_freq=args.min_freq)
    if not freq_raw:
        print(f"  ❌ 没有任何 attr 词通过筛选, 退出"); return

    # ── STEP 2.5: ★ v3 多层启发式过滤 ──
    print(f"\n{'-'*60}\nSTEP 2.5 | v3 多层启发式过滤 (数字/特殊字符/专有名词/序数/超长)\n{'-'*60}")
    freq_after_heur = filter_advanced_noise(freq_raw, max_words=args.max_words_per_attr)
    if not freq_after_heur:
        print(f"  ❌ 过滤后没剩词, 退出"); return

    # ── STEP 2.6: ★ v3 caption 词频过滤 ──
    print(f"\n{'-'*60}\nSTEP 2.6 | v3 COCO caption 词频过滤\n{'-'*60}")
    if args.caption_freq_threshold > 0:
        caption_freq = build_caption_word_freq(args.coco_caption_json)
        freq_after_cap = filter_by_caption_freq(
            freq_after_heur, caption_freq, args.caption_freq_threshold)
    else:
        print(f"  --caption_freq_threshold=0, 跳过 caption 词频过滤")
        caption_freq = None
        freq_after_cap = freq_after_heur
    if not freq_after_cap:
        print(f"  ❌ 过滤后没剩词, 退出"); return

    # ── STEP 3: VAW 优先去重 ──
    print(f"\n{'-'*60}\nSTEP 3 | VAW 优先去重\n{'-'*60}")
    freq = deduplicate_against_vaw(freq_after_cap, vaw_attr_set)
    if not freq:
        print(f"  ❌ 去重后没剩词, 退出"); return
    attr_words = sorted(freq.keys(), key=lambda a: -freq[a])

    # ── STEP 4: SentenceBERT ──
    print(f"\n{'-'*60}\nSTEP 4 | 加载 SentenceBERT\n{'-'*60}")
    model = load_sentence_bert(args.model_path)

    # ── STEP 5: 算 dim 锚点 ──
    print(f"\n{'-'*60}\nSTEP 5 | 算 dim 锚点 embedding\n{'-'*60}")
    dim_names, centroids = build_dim_centroids(model, DIM_ANCHORS)

    # ── STEP 6: 分类 ──
    print(f"\n{'-'*60}\nSTEP 6 | 分类\n{'-'*60}")
    dim_to_attrs = classify_attrs(model, dim_names, centroids, attr_words,
                                   sim_threshold=args.sim_threshold,
                                   batch_size=args.batch_size)

    # ── STEP 7: ★ v3 per-dim 数量上限 ──
    print(f"\n{'-'*60}\nSTEP 7 | v3 per-dim 数量上限\n{'-'*60}")
    dim_to_attrs = cap_per_dim(dim_to_attrs, freq, max_per_dim=args.max_per_dim)

    # ── STEP 7.5: ★ v3 可选丢 misc ──
    if args.no_misc:
        print(f"\n{'-'*60}\nSTEP 7.5 | v3 丢弃 misc dim\n{'-'*60}")
        dim_to_attrs = remove_misc_dim(dim_to_attrs)

    # ── STEP 8: 概览 + 保存 ──
    print_summary(dim_to_attrs, freq)
    save_output(dim_to_attrs, freq, args.output_json)

    print(f"\n{SEP}")
    print(f"  ✅ v3 完成!")
    print(f"  ★ v3 过滤汇总:")
    print(f"    [STEP 2]    VG 频次过滤 (>= {args.min_freq})")
    print(f"    [STEP 2.5]  v3 启发式过滤 (数字/特殊字符/专有名词/序数/超长)")
    print(f"    [STEP 2.6]  v3 caption 词频过滤 (>= {args.caption_freq_threshold})")
    print(f"    [STEP 3]    VAW 优先去重")
    print(f"    [STEP 6]    SentenceBERT cos sim (>= {args.sim_threshold})")
    print(f"    [STEP 7]    per-dim cap (<= {args.max_per_dim})")
    if args.no_misc:
        print(f"    [STEP 7.5]  丢弃 misc dim")
    print(f"\n  下一步: 跑 build_attr_memory.py v6.9.1 (会自动用此 vg_attribute_types.json)")
    print(f"{SEP}\n")


if __name__ == "__main__":
    main()