#!/usr/bin/env python3
"""
constants.py ── 常量、词汇表、COCO映射

提供给 caption_model 包内其他模块导入。
"""

# ============================================================
# 维度常量
# ============================================================
NUM_BINS = 1000
QFORMER_HIDDEN = 768
EVA01_DIM = 1408
RETRIEVAL_DIM = 768
LLM_HIDDEN = 4096

# ============================================================
# 检索阈值
# ============================================================
NAME_SIM_THRESHOLD = 0.15

# ============================================================
# 无效 tag 集合
# ============================================================
INVALID_TAG_NAMES_EXTENDED = {
    "object", "scene", "region", "unknown", "background", "stuff",
    "", "other", "thing", "item", "entity", "none", "null",
    "objects", "things", "items", "regions", "scenes",
}

INVALID_TAG_NAMES = {
    "object", "scene", "region", "unknown", "background", "stuff",
}

# ============================================================
# COCO 名称集合
# ============================================================
COCO_NAME_SET = {
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
    "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl",
    "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza",
    "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet",
    "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven",
    "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
    "teddy bear", "hair drier", "toothbrush",
}

# ============================================================
# COCO ID → 名称
# ============================================================
COCO_ID_TO_NAME = {
    1: "person", 2: "bicycle", 3: "car", 4: "motorcycle", 5: "airplane",
    6: "bus", 7: "train", 8: "truck", 9: "boat", 10: "traffic light",
    11: "fire hydrant", 13: "stop sign", 14: "parking meter", 15: "bench",
    16: "bird", 17: "cat", 18: "dog", 19: "horse", 20: "sheep", 21: "cow",
    22: "elephant", 23: "bear", 24: "zebra", 25: "giraffe",
    27: "backpack", 28: "umbrella", 31: "handbag", 32: "tie", 33: "suitcase",
    34: "frisbee", 35: "skis", 36: "snowboard", 37: "sports ball", 38: "kite",
    39: "baseball bat", 40: "baseball glove", 41: "skateboard", 42: "surfboard",
    43: "tennis racket", 44: "bottle", 46: "wine glass", 47: "cup", 48: "fork",
    49: "knife", 50: "spoon", 51: "bowl", 52: "banana", 53: "apple",
    54: "sandwich", 55: "orange", 56: "broccoli", 57: "carrot", 58: "hot dog",
    59: "pizza", 60: "donut", 61: "cake", 62: "chair", 63: "couch",
    64: "potted plant", 65: "bed", 67: "dining table", 70: "toilet",
    72: "tv", 73: "laptop", 74: "mouse", 75: "remote", 76: "keyboard",
    77: "cell phone", 78: "microwave", 79: "oven", 80: "toaster", 81: "sink",
    82: "refrigerator", 84: "book", 85: "clock", 86: "vase", 87: "scissors",
    88: "teddy bear", 89: "hair drier", 90: "toothbrush",
}

# ============================================================
# 数字单词 prompt
# ============================================================
NUM_WORDS_PROMPT = {
    1: "", 2: "two ", 3: "three ", 4: "four ",
    5: "five ", 6: "six ", 7: "seven ", 8: "eight ",
    9: "nine ", 10: "ten ", 11: "eleven ", 12: "twelve ",
}


def num_prefix(count: int) -> str:
    return NUM_WORDS_PROMPT.get(count, f"{count} ")


# ============================================================
# 复数化映射
# ============================================================
_PLURAL_MAP = {
    "person": "people", "child": "children", "man": "men", "woman": "women",
    "knife": "knives", "bus": "buses", "mouse": "mice", "sheep": "sheep",
}


def pluralize(name: str) -> str:
    if name in _PLURAL_MAP:
        return _PLURAL_MAP[name]
    if name.endswith(('s', 'x', 'ch', 'sh')):
        return name + 'es'
    if name.endswith('y') and len(name) > 1 and name[-2] not in 'aeiou':
        return name[:-1] + 'ies'
    return name + 's'
