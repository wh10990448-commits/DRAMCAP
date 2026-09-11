#!/usr/bin/env python3
"""
train.py  v6.2.1 (Karpathy 118k 适配版)
=================================================================
适配: model.py v6.6.1 + caption_model 包 + build_attr_memory.py v6.8
基础: 完全继承 v6.0 (v6.8-attr 兼容版)

v6.2.1 改动 (相对 v6.0): ★ 仅 3 处, 其他不变
═════════════════════════════════════════════════════════════════
  [KARPATHY-DS]  OnlineCaptionDataset 自动识别两种 caption 格式:
                   - list  → Karpathy 扁平 [{caption, image, image_id}, ...]
                   - dict  → COCO 原版 {images: [...], annotations: [...]}
                 内部统一存成同一份 self.captions[numeric_id] 结构
                 'coco_546188' 类 image_id 自动剥掉 'coco_' 前缀拿到数字 ID

  [DUAL-INST]    CocoBoxLoader.__init__ 加 val_json_path 参数 (默认 None)
                 if val_json_path: 把 instances_val2014.json 合并进 _image_anns
                 train2014 / val2014 的 image_id 不重叠, 不会冲突

  [MAIN-DISPATCH] main() 数据集路径用 4 级优先级解析 (--train_json 优先):
                     1. args.train_json                          (CLI, run.sh 传)
                     2. args.caption_file                        (CLI, 旧用法)
                     3. config.paths.coco_karpathy_train_json    (config.yaml V6.5 字段)
                     4. config.paths.coco_train_captions         (COCO 原版 fallback)
                   image_dir 从 config.paths.coco_train_images 取
                     (V6.5 该字段默认改为父目录, 含 train2014/ + val2014/)
                   不需要 dataset_format 开关 — OnlineCaptionDataset 自动识别格式

继承 v6.0 (无功能性变化, 全部保留):
  [V6.8-LOG]    main() 里加载 PKL 后的 attr_memory.pkl 概要日志
  [V6.8-RESUME] --resume 路径的 mismatch 智能分类打印
  [ITG-LOSS]    Loss = 标准 ITG, 仅一个 NTP cross-entropy
  [PER-Q-RETR]  样本诊断的 per-query 检索打印
  [5GT-EXPAND]  每张图 × 5 条 GT 展开
  [LS-CUR]      Label Smoothing Curriculum
  [v5.5.0]      辅助 loss 全置零 / 生成参数 / optimizer 自动覆盖
"""

import os
import json
import argparse
import logging
import random
import math
import re
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from losses import CaptionLoss

# ── CUDA / autocast 初始化 ─────────────────────────────────────
if torch.cuda.is_available():
    try:
        _BF16_OK = torch.cuda.is_bf16_supported()
        _dummy   = torch.zeros(1, device='cuda:0')
        del _dummy
    except Exception:
        _BF16_OK = False
else:
    _BF16_OK = False


def _safe_autocast():
    if _BF16_OK:
        return torch.amp.autocast('cuda', dtype=torch.bfloat16)
    return torch.cuda.amp.autocast(dtype=torch.float16)


from config import get_config, get_ablation_config, create_directories, print_config
from caption_model import CaptionModel, _clean_caption

try:
    from preprocess_features_final import (
        VerbEnhancer, _interact_from_objects, _INTERACT_RULES,
    )
    from attribute_enhancer import AttributeIndexLoader
    _VERB_ENHANCER_AVAILABLE = True
except ImportError:
    _VERB_ENHANCER_AVAILABLE = False
    AttributeIndexLoader     = None
    VerbEnhancer             = None


# ─────────────────────────────────────────────────────────────
# ★ [V6.2.1 NEW] Karpathy image_id 解析: 'coco_546188' → '546188'
# ─────────────────────────────────────────────────────────────
def _parse_karpathy_image_id(s):
    """
    支持 3 种 image_id:
      - 'coco_546188' (Karpathy 标准, 剥掉 'coco_' 前缀)
      - '546188'      (纯数字字符串)
      - 546188        (int)
    返回数字字符串 (跟 COCO instances JSON 里 image_id 字段对齐)
    """
    if isinstance(s, int):
        return str(s)
    s = str(s).strip()
    if s.startswith('coco_'):
        return s[5:]
    return s


# ─────────────────────────────────────────────────────────────
# 辅助：过滤无效标签
# ─────────────────────────────────────────────────────────────
_INVALID_DISPLAY_TAGS = {
    "object", "scene", "region", "unknown",
    "background", "stuff", ""
}


def _filter_real_tags(tag_list, max_show=5):
    if not tag_list:
        return []
    from collections import Counter
    counts = Counter()
    for t in tag_list:
        t_str = str(t).strip().lower()
        if t_str not in _INVALID_DISPLAY_TAGS:
            counts[str(t).strip()] += 1
    result = []
    for name, count in sorted(counts.items(), key=lambda x: -x[1])[:max_show]:
        if count >= 2:
            result.append(f"{name}×{count}")
        else:
            result.append(name)
    return result


def _fmt_name_attrs(raw_names_batch, attrs_batch, max_names=4):
    if not raw_names_batch:
        return ""
    parts = []
    for name in raw_names_batch[:max_names]:
        if not name:
            continue
        attr_list = attrs_batch.get(name, [])
        if attr_list:
            if isinstance(attr_list[0], dict):
                vals = [a.get('value', '') for a in attr_list[:3] if a.get('value')]
            else:
                vals = [str(a) for a in attr_list[:3] if a]
            if vals:
                parts.append(f"{name}({','.join(vals)})")
                continue
        parts.append(name)
    return ", ".join(parts)


def _fmt_attr_details(raw_names_batch, attrs_batch, max_names=4):
    if not raw_names_batch:
        return ""
    attr_parts = []
    for name in raw_names_batch[:max_names]:
        if not name:
            continue
        a_list = attrs_batch.get(name, [])
        if not a_list:
            continue
        if isinstance(a_list[0], dict):
            vals = [f"{a.get('dim','?')}={a.get('value','')}"
                    for a in a_list[:3] if a.get('value')]
        else:
            vals = [str(a) for a in a_list[:3] if a]
        if vals:
            attr_parts.append(f"{name}:[{','.join(vals)}]")
    return "; ".join(attr_parts)


def _norm_name(s: str) -> str:
    return s.lower().strip().replace('_', ' ').replace('-', ' ')


def _roi_hit_check(roi_name: str, gt_caption: str) -> str:
    if not roi_name or not gt_caption:
        return "❌"
    caption_words = set(re.findall(r'[a-z]+', gt_caption.lower()))
    roi_words     = re.findall(r'[a-z]+', roi_name.lower())
    if not roi_words:
        return "❌"
    stopwords = {'a','an','the','of','in','on','at','to','and','or','is','are'}
    roi_content = [w for w in roi_words if w not in stopwords] or roi_words
    matched = sum(1 for w in roi_content if w in caption_words)
    if matched == len(roi_content):
        return "✅"
    elif matched > 0:
        return "⚠️"
    return "❌"


def _roi_hit_best(roi_name: str, gt_captions) -> str:
    if isinstance(gt_captions, str):
        gt_captions = [gt_captions]
    best = "❌"
    for cap in gt_captions:
        mark = _roi_hit_check(roi_name, cap)
        if mark == "✅":
            return "✅"
        if mark == "⚠️":
            best = "⚠️"
    return best


def _format_retrieval_diag(vote_topk, vote_denoised, gt_captions, max_show=8):
    lines = []
    if vote_topk:
        raw_tagged = []
        for item in vote_topk[:max_show]:
            if isinstance(item, (list, tuple)) and len(item) == 2:
                name, sim = item
                sim_str = f"{sim:.2f}" if isinstance(sim, float) else str(sim)
            else:
                name, sim_str = str(item), ""
            mark = _roi_hit_best(name, gt_captions)
            raw_tagged.append(f"{name}({sim_str}){mark}" if sim_str else f"{name}{mark}")
        lines.append(f"    ├─ 检索(Step1): [{', '.join(raw_tagged)}]")
    else:
        lines.append("    ├─ 检索(Step1): [无]")
    if vote_denoised:
        dn_valid  = [n for n in vote_denoised if n and n.strip()][:max_show]
        dn_tagged = []
        for name in dn_valid:
            mark = _roi_hit_best(name, gt_captions)
            dn_tagged.append(f"{name}{mark}")
        lines.append(f"    └─ 注入(Step2): [{', '.join(dn_tagged)}]  共{len(dn_valid)}个")
    else:
        lines.append("    └─ 注入(Step2): [无]")
    return "\n".join(lines)


def _get_token_layout(model) -> str:
    """[V6.8.1] soft_retriever 已删, layout 简化为只显示 vis token 数"""
    return "32vis (硬检索: name_8 + attr_6)"


# ─────────────────────────────────────────────────────────────
# YOLOe Detector (可选,默认不用)
# ─────────────────────────────────────────────────────────────
class YOLOeDetector:
    def __init__(self, model_path, conf_threshold=0.25, max_rois=13, core_count=4):
        self.model_path     = model_path
        self.conf_threshold = conf_threshold
        self.max_rois       = max_rois
        self.core_count     = core_count
        self.model          = None
        self.is_loaded      = False
        if not model_path or not os.path.exists(model_path):
            print(f"[YOLOe] ⚠️ 模型路径不存在: {model_path}")
            return
        try:
            self._load_model(model_path)
        except Exception as e:
            print(f"[YOLOe] ❌ 加载失败: {e}")

    def _load_model(self, model_path):
        import time
        t0 = time.time()
        print(f"[YOLOe] 加载: {model_path}  conf≥{self.conf_threshold}")
        try:
            from ultralytics import YOLO
            if os.path.isdir(model_path):
                pt_candidates = [os.path.join(model_path, 'best.pt'),
                                 os.path.join(model_path, 'last.pt')]
                weights_dir = os.path.join(model_path, 'weights')
                if os.path.isdir(weights_dir):
                    pt_candidates.insert(0, os.path.join(weights_dir, 'best.pt'))
                    pt_candidates.insert(1, os.path.join(weights_dir, 'last.pt'))
                for root, dirs, files in os.walk(model_path):
                    for f in files:
                        if f.endswith('.pt'):
                            pt_candidates.append(os.path.join(root, f))
                pt_path = None
                for p in pt_candidates:
                    if os.path.exists(p): pt_path = p; break
                if pt_path is None:
                    print(f"[YOLOe] ⚠️ 目录中未找到 .pt: {model_path}"); return
            else:
                pt_path = model_path
            self.model = YOLO(pt_path)
            elapsed = time.time() - t0
            num_classes = len(self.model.names) if hasattr(self.model, 'names') else 0
            print(f"[YOLOe] ✅ 加载完成: {pt_path} | {num_classes}类 ({elapsed:.1f}s)")
            self.is_loaded = True
        except ImportError:
            print("[YOLOe] ⚠️ ultralytics 未安装，尝试 torch.load...")
            try:
                pt = (model_path if not os.path.isdir(model_path)
                      else os.path.join(model_path, 'best.pt'))
                self.model = torch.load(pt, map_location='cpu', weights_only=False)
                self.is_loaded = True
                elapsed = time.time() - t0
                print(f"[YOLOe] ✅ torch.load 完成 ({elapsed:.1f}s)")
            except Exception as e2:
                print(f"[YOLOe] ❌ 所有加载方式均失败: {e2}")
        except Exception as e:
            print(f"[YOLOe] ❌ 加载异常: {e}")

    @torch.no_grad()
    def detect(self, image, max_rois=None, core_count=None, max_per_cat=2):
        if not self.is_loaded or self.model is None: return None
        max_r  = max_rois   or self.max_rois
        core_c = core_count or self.core_count
        try:
            results = self.model(image, conf=self.conf_threshold, verbose=False)
            if not results or len(results) == 0: return None
            result = results[0]
            if result.boxes is None or len(result.boxes) == 0: return None
            if isinstance(image, Image.Image): W, H = image.size
            elif hasattr(result, 'orig_shape'): H, W = result.orig_shape[:2]
            else: W, H = 640, 480
            xyxy = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            cls_ids = result.boxes.cls.cpu().numpy().astype(int)
            names_map = getattr(self.model, 'names', {})
            areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
            order = np.argsort(-areas)
            boxes = []; cat_count = {}; overflow = []
            for idx in order:
                x1_abs, y1_abs, x2_abs, y2_abs = xyxy[idx]
                cid  = int(cls_ids[idx]); conf = float(confs[idx])
                name = names_map.get(cid, f'class_{cid}')
                x1 = max(0.0, x1_abs / W); y1 = max(0.0, y1_abs / H)
                x2 = min(1.0, x2_abs / W); y2 = min(1.0, y2_abs / H)
                if x2 - x1 < 0.01 or y2 - y1 < 0.01: continue
                cnt = cat_count.get(cid, 0)
                box_dict = {'bbox':[x1,y1,x2,y2],'category_id':cid,'name':name,
                            'confidence':conf,'is_core':len(boxes)<core_c}
                if cnt < max_per_cat:
                    boxes.append(box_dict); cat_count[cid] = cnt + 1
                    if len(boxes) >= max_r: break
                else: overflow.append(box_dict)
            if len(boxes) < max_r:
                for box_dict in overflow:
                    if len(boxes) >= max_r: break
                    box_dict['is_core'] = len(boxes) < core_c
                    boxes.append(box_dict)
            return boxes if boxes else None
        except Exception:
            return None


# ─────────────────────────────────────────────────────────────
# CocoBoxLoader  [V6.2.1: 加 val_json_path 双 instances 合并]
# ─────────────────────────────────────────────────────────────
class CocoBoxLoader:
    COCO_ID_TO_NAME = {
        1:"person",2:"bicycle",3:"car",4:"motorcycle",5:"airplane",6:"bus",
        7:"train",8:"truck",9:"boat",10:"traffic light",11:"fire hydrant",
        13:"stop sign",14:"parking meter",15:"bench",16:"bird",17:"cat",
        18:"dog",19:"horse",20:"sheep",21:"cow",22:"elephant",23:"bear",
        24:"zebra",25:"giraffe",27:"backpack",28:"umbrella",31:"handbag",
        32:"tie",33:"suitcase",34:"frisbee",35:"skis",36:"snowboard",
        37:"sports ball",38:"kite",39:"baseball bat",40:"baseball glove",
        41:"skateboard",42:"surfboard",43:"tennis racket",44:"bottle",
        46:"wine glass",47:"cup",48:"fork",49:"knife",50:"spoon",51:"bowl",
        52:"banana",53:"apple",54:"sandwich",55:"orange",56:"broccoli",
        57:"carrot",58:"hot dog",59:"pizza",60:"donut",61:"cake",62:"chair",
        63:"couch",64:"potted plant",65:"bed",67:"dining table",70:"toilet",
        72:"tv",73:"laptop",74:"mouse",75:"remote",76:"keyboard",
        77:"cell phone",78:"microwave",79:"oven",80:"toaster",81:"sink",
        82:"refrigerator",84:"book",85:"clock",86:"vase",87:"scissors",
        88:"teddy bear",89:"hair drier",90:"toothbrush",
    }

    def __init__(self, json_path, val_json_path=None, max_rois=13, core_count=4):
        """
        Args:
            json_path:     train instances JSON (instances_train2014.json)
            val_json_path: ★ V6.2.1 新加: val instances JSON (instances_val2014.json)
                           Karpathy split train.json 含 val2014/ 路径, 必须合并
        """
        self.max_rois = max_rois; self.core_count = core_count
        self.is_loaded = False
        self._image_anns = {}; self._image_sizes = {}
        self._cat_names = {}; self.num_categories = 0
        if not json_path or not os.path.exists(json_path):
            print(f"[CocoBox] ⚠️ 文件不存在: {json_path}"); return
        import time; t0 = time.time()

        # ── 1. 加载 train instances ──
        print(f"[CocoBox] 加载 train: {json_path}")
        with open(json_path) as f: data = json.load(f)
        for cat in data.get('categories', []):
            self._cat_names[cat['id']] = cat['name']
        self.num_categories = len(self._cat_names)
        for img in data.get('images', []):
            self._image_sizes[str(img['id'])] = (img.get('width',640), img.get('height',480))
        n_train_ann = 0
        for ann in data.get('annotations', []):
            if ann.get('iscrowd', 0): continue
            bbox = ann.get('bbox', [])
            if len(bbox) != 4 or bbox[2] < 1 or bbox[3] < 1: continue
            img_id = str(ann['image_id'])
            if img_id not in self._image_anns: self._image_anns[img_id] = []
            self._image_anns[img_id].append(ann)
            n_train_ann += 1
        n_train_img = len(self._image_anns)

        # ── 2. ★ [V6.2.1 NEW] 合并 val instances (Karpathy split 需要) ──
        n_val_ann = 0; n_val_img = 0
        if val_json_path and os.path.exists(val_json_path):
            print(f"[CocoBox] 加载 val:   {val_json_path}")
            with open(val_json_path) as f: val_data = json.load(f)
            # 类别表合并 (一般 train/val 类别相同, 求并集兜底)
            for cat in val_data.get('categories', []):
                if cat['id'] not in self._cat_names:
                    self._cat_names[cat['id']] = cat['name']
            # 图像 sizes 合并 (train2014 / val2014 image_id 不重叠)
            for img in val_data.get('images', []):
                self._image_sizes[str(img['id'])] = (img.get('width',640), img.get('height',480))
            # 标注合并
            img_id_seen_val = set()
            for ann in val_data.get('annotations', []):
                if ann.get('iscrowd', 0): continue
                bbox = ann.get('bbox', [])
                if len(bbox) != 4 or bbox[2] < 1 or bbox[3] < 1: continue
                img_id = str(ann['image_id'])
                if img_id not in self._image_anns: self._image_anns[img_id] = []
                self._image_anns[img_id].append(ann)
                n_val_ann += 1
                img_id_seen_val.add(img_id)
            n_val_img = len(img_id_seen_val)
            self.num_categories = len(self._cat_names)
        elif val_json_path:
            print(f"[CocoBox] ⚠️ val instances 不存在 (Karpathy 部分图无 box): {val_json_path}")

        # ── 3. 总结日志 ──
        elapsed = time.time() - t0
        total_ann = n_train_ann + n_val_ann
        total_img = len(self._image_anns)
        print(f"[CocoBox] ✅ {self.num_categories}类 | {total_img}图 | {total_ann}标注 ({elapsed:.1f}s)")
        if n_val_ann > 0:
            print(f"[CocoBox]    train: {n_train_img}图/{n_train_ann}标注  + "
                  f"val: {n_val_img}图/{n_val_ann}标注 [V6.2.1 DUAL-INST]")
        self.is_loaded = True

    def has_image(self, img_id): return str(img_id) in self._image_anns

    def get_image_rois(self, img_id, max_rois=None, core_count=None, max_per_cat=2):
        img_id = str(img_id)
        if img_id not in self._image_anns: return None
        max_r = max_rois or self.max_rois
        core_c = core_count or self.core_count
        W, H = self._image_sizes.get(img_id, (640, 480))
        anns = self._image_anns[img_id]
        anns_sorted = sorted(anns, key=lambda a: a['bbox'][2]*a['bbox'][3], reverse=True)

        def _to_box(ann, is_core_flag):
            x, y, w, h = ann['bbox']
            x1 = max(0.0, x/W); y1 = max(0.0, y/H)
            x2 = min(1.0, (x+w)/W); y2 = min(1.0, (y+h)/H)
            if x2-x1<0.01 or y2-y1<0.01: return None
            cid = ann.get('category_id', 0)
            name = self._cat_names.get(cid, self.COCO_ID_TO_NAME.get(cid,'object'))
            return {'bbox':[x1,y1,x2,y2],'category_id':cid,'name':name,
                    'confidence':1.0,'is_core':is_core_flag}

        boxes = []; cat_count = {}; overflow = []
        for ann in anns_sorted:
            cid = ann.get('category_id', 0)
            cnt = cat_count.get(cid, 0)
            if cnt < max_per_cat:
                box = _to_box(ann, len(boxes)<core_c)
                if box:
                    boxes.append(box); cat_count[cid] = cnt + 1
                    if len(boxes) >= max_r: break
            else:
                overflow.append(ann)
        if len(boxes) < max_r:
            for ann in overflow:
                if len(boxes) >= max_r: break
                box = _to_box(ann, len(boxes)<core_c)
                if box: boxes.append(box)
        return boxes if boxes else None


# ─────────────────────────────────────────────────────────────
# Dataset  [5GT-EXPAND]  [V6.2.1: 自动识别 Karpathy / COCO 两种格式]
# ─────────────────────────────────────────────────────────────
class OnlineCaptionDataset(Dataset):
    def __init__(self, image_dir, caption_file, tokenizer,
                 max_len=64, num_captions_per_image=5,
                 lvis_loader=None, yoloe_detector=None,
                 max_rois=13, core_count=4):
        self.image_dir = image_dir
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.num_captions = num_captions_per_image
        self.max_rois = max_rois
        self.core_count = core_count
        self.box_loader = lvis_loader
        self.yoloe_detector = yoloe_detector
        self.current_epoch = 0

        with open(caption_file) as f:
            data = json.load(f)

        self.captions = {}
        self.id_to_file = {}

        # ★ [V6.2.1 NEW] 自动识别格式 ─────────────────────────────────
        if isinstance(data, list):
            # Karpathy 扁平: [{caption, image, image_id}, ...]
            # image:    'val2014/COCO_val2014_xxx.jpg' 或 'train2014/COCO_train2014_xxx.jpg'
            # image_id: 'coco_546188' 或 '546188' 或 546188
            self._format = 'karpathy'
            for entry in data:
                if not isinstance(entry, dict): continue
                raw_id = entry.get('image_id', entry.get('image', ''))
                numeric_id = _parse_karpathy_image_id(raw_id)
                if not numeric_id: continue
                if numeric_id not in self.captions:
                    self.captions[numeric_id] = []
                cap = entry.get('caption', '')
                if cap: self.captions[numeric_id].append(cap)
                # image 字段含 train2014/ 或 val2014/ 前缀, 是相对路径
                img_path_rel = entry.get('image', '')
                if img_path_rel and numeric_id not in self.id_to_file:
                    self.id_to_file[numeric_id] = img_path_rel
            print(f"[ONLINE-5GT] 格式: Karpathy 扁平 ({len(data)} 条记录)")
        elif isinstance(data, dict) and 'annotations' in data:
            # COCO 原版: {images: [...], annotations: [{image_id, caption}, ...]}
            self._format = 'coco'
            for ann in data['annotations']:
                k = str(ann['image_id'])
                if k not in self.captions: self.captions[k] = []
                self.captions[k].append(ann['caption'])
            if 'images' in data:
                for img_info in data['images']:
                    self.id_to_file[str(img_info['id'])] = img_info['file_name']
            print(f"[ONLINE-5GT] 格式: COCO 原版 (annotations: {len(data['annotations'])})")
        else:
            raise ValueError(
                f"未知 caption 文件格式: {caption_file}\n"
                f"期望: list of {{caption, image, image_id}} (Karpathy) "
                f"或 dict with 'annotations' key (COCO)")
        # ────────────────────────────────────────────────────────────

        self.img_ids = []
        missing = 0
        for img_id in self.captions:
            p = self._get_image_path(img_id)
            if p and os.path.exists(p): self.img_ids.append(img_id)
            else: missing += 1

        # [5GT-EXPAND] 展开为 (img_id, caption_index) 独立样本
        self.samples = []
        for img_id in self.img_ids:
            caps = self.captions[img_id]
            n_caps = min(len(caps), num_captions_per_image)
            for ci in range(n_caps):
                self.samples.append((img_id, ci))

        print(f"[ONLINE-5GT] 有效图像: {len(self.img_ids)}, 缺失: {missing}")
        print(f"[ONLINE-5GT] 展开样本: {len(self.samples)} "
              f"(每图×{num_captions_per_image} GT)")

        lvis_hit = sum(1 for iid in self.img_ids
                       if self.box_loader and self.box_loader.has_image(iid))
        if self.box_loader and getattr(self.box_loader, 'is_loaded', False):
            pct = 100 * lvis_hit / max(len(self.img_ids), 1)
            print(f"[CocoBox] 覆盖: {lvis_hit}/{len(self.img_ids)} ({pct:.1f}%)")

    def _get_image_path(self, img_id):
        if img_id in self.id_to_file:
            return os.path.join(self.image_dir, self.id_to_file[img_id])
        for fmt in [f"{int(img_id):012d}.jpg", f"{img_id}.jpg"]:
            p = os.path.join(self.image_dir, fmt)
            if os.path.exists(p): return p
        return None

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        img_id, cap_idx = self.samples[idx]
        img_path = self._get_image_path(img_id)
        image = Image.open(img_path).convert('RGB')
        all_caps = self.captions[img_id]
        cap = all_caps[cap_idx] if cap_idx < len(all_caps) else all_caps[0]
        enc = self.tokenizer(cap, max_length=self.max_len,
                             padding='max_length', truncation=True,
                             return_tensors='pt')
        lvis_rois = None
        if self.box_loader and self.box_loader.has_image(img_id):
            lvis_rois = self.box_loader.get_image_rois(
                img_id, max_rois=self.max_rois, core_count=self.core_count)
        elif self.yoloe_detector and self.yoloe_detector.is_loaded:
            try:
                lvis_rois = self.yoloe_detector.detect(
                    image, max_rois=self.max_rois, core_count=self.core_count)
            except Exception:
                lvis_rois = None
        return {
            'img_id': img_id,
            'image': image,
            'text_tokens': enc['input_ids'].squeeze(0),
            'text_mask': enc['attention_mask'].squeeze(0).bool(),
            'all_captions': all_caps,
            'lvis_rois': lvis_rois,
        }


def online_collate_fn(batch):
    return {
        'img_ids':      [b['img_id'] for b in batch],
        'images':       [b['image'] for b in batch],
        'text_tokens':  torch.stack([b['text_tokens'] for b in batch]),
        'text_mask':    torch.stack([b['text_mask'] for b in batch]),
        'all_captions': [b.get('all_captions', []) for b in batch],
        'lvis_rois':    [b['lvis_rois'] for b in batch],
    }


# ─────────────────────────────────────────────────────────────
# 生成参数
# ─────────────────────────────────────────────────────────────
_SAMPLE_GEN_KWARGS = dict(
    max_new_tokens=30, min_new_tokens=8, num_beams=5,
    length_penalty=1.0, repetition_penalty=1.0,
    no_repeat_ngram_size=3, early_stopping=True,
)
_EVAL_GEN_KWARGS = dict(
    max_new_tokens=60, min_new_tokens=8, num_beams=5,
    length_penalty=1.0, repetition_penalty=1.0,
    no_repeat_ngram_size=3, early_stopping=True,
)


# ─────────────────────────────────────────────────────────────
# [V6.8] 续训 mismatch 智能分类
# ─────────────────────────────────────────────────────────────
def _classify_resume_mismatch(missing, unexpected, logger):
    """
    把 strict=False 加载的 mismatch 分成两类:
      [安全]      v6.8 已知会变化的 (soft_retriever.stream_bias / stream_weights)
      [需警惕]    其他不在白名单的 mismatch
    """
    _V68_SAFE_PATTERNS = [
        'soft_retriever.stream_bias',
        'soft_retriever.stream_weights',
    ]
    _V65_DEPRECATED_PATTERNS = [
        'attr_qformer.',
    ]

    def _is_safe(k):
        return any(p in k for p in _V68_SAFE_PATTERNS + _V65_DEPRECATED_PATTERNS)

    safe_missing    = [k for k in missing    if _is_safe(k)]
    safe_unexpected = [k for k in unexpected if _is_safe(k)]
    risky_missing    = [k for k in missing    if not _is_safe(k)]
    risky_unexpected = [k for k in unexpected if not _is_safe(k)]

    logger.info(f"  [续训-mismatch 分类]")
    logger.info(f"    [V6.8 安全] missing    ({len(safe_missing)}):    "
                f"{safe_missing[:3]}{'...' if len(safe_missing) > 3 else ''}")
    logger.info(f"    [V6.8 安全] unexpected ({len(safe_unexpected)}): "
                f"{safe_unexpected[:3]}{'...' if len(safe_unexpected) > 3 else ''}")
    if risky_missing:
        logger.warning(f"    [⚠️  需警惕] missing    ({len(risky_missing)}): "
                       f"{risky_missing[:5]}")
    else:
        logger.info(f"    [✅ 无其他] missing 全部在 v6.8 白名单内")
    if risky_unexpected:
        logger.warning(f"    [⚠️  需警惕] unexpected ({len(risky_unexpected)}): "
                       f"{risky_unexpected[:5]}")
    else:
        logger.info(f"    [✅ 无其他] unexpected 全部在 v6.8 白名单内")


# ─────────────────────────────────────────────────────────────
# [V6.8] 打印 attr_memory.pkl 概要
# ─────────────────────────────────────────────────────────────
def _log_attr_memory_summary(attr_pkl_path, logger):
    """读 attr_memory.pkl 主要字段, 兼容 v6.5/v6.7/v6.8"""
    if not attr_pkl_path or not os.path.exists(attr_pkl_path):
        return
    try:
        import pickle
        from collections import Counter
        with open(attr_pkl_path, 'rb') as f:
            am = pickle.load(f)
        version = am.get('version', '?')
        n_vectors = am.get('n_vectors', None)
        n_objects = am.get('n_objects', None)
        n_dims = am.get('n_dims', None)
        n_unified = am.get('n_unified_attr_table', None)
        sources = am.get('sources', [])

        logger.info(f"  [ATTR-PKL] version={version}")
        if n_vectors is not None:
            logger.info(f"  [ATTR-PKL] n_vectors={n_vectors:,}  "
                        f"n_objects={n_objects}")
        if n_dims is not None:
            logger.info(f"  [ATTR-PKL] [V6.8] 入库 dim 数: {n_dims}")
        if n_unified is not None:
            logger.info(f"  [ATTR-PKL] [V6.8] unified attr→dim 表: "
                        f"{n_unified:,} attr 词")
        if sources:
            sc = Counter(sources)
            parts = [f"{s}={sc.get(s,0)}({sc.get(s,0)/len(sources)*100:.1f}%)"
                     for s in ['PACO','VAW','VG'] if sc.get(s,0)>0]
            if parts:
                logger.info(f"  [ATTR-PKL] 三源占比: {' + '.join(parts)}")
    except Exception as e:
        logger.warning(f"  [ATTR-PKL] 概要读取失败: {e}")


# ─────────────────────────────────────────────────────────────
# Trainer  v6.0  (无任何改动)
# ─────────────────────────────────────────────────────────────
class Trainer:
    def __init__(self, config, model, loss_fn, train_loader, logger,
                 online_mode=False, label_smoothing=0.0,
                 show_retrieval_diag=True):
        self.config       = config
        self.model        = model.to(config.training.device)
        self.loss_fn      = loss_fn
        self.train_loader = train_loader
        self.logger       = logger
        self.device       = torch.device(config.training.device)
        self.tokenizer    = None
        self.stage        = 2
        self.epoch        = 0
        self.global_step  = 0
        self.online_mode  = online_mode
        self.print_samples_every = getattr(
            config.training, 'print_samples_every', 200)
        self._vision_loaded     = False
        self._evaclip_init_done = False

        self.label_smoothing = label_smoothing

        # 辅助 loss 全部置零
        self.w_ntp          = 1.0
        self.w_contrast     = 0.0
        self.w_ret_align    = 0.0
        self.w_name_soft1   = 0.0
        self.w_name_stage2  = 0.0
        self.ret_align_warmup_epochs = 0
        self.w_ret_align_min = 0.0
        self.contrast_warmup_steps = 1
        self.unfreeze_new_modules_epoch = getattr(
            config.training, 'unfreeze_new_modules_epoch', 99)
        self.use_ms_eva = getattr(
            getattr(config, 'ablation', None), 'use_ms_eva', True)

        # [LS-CUR] Label Smoothing Curriculum (默认关闭)
        self.ls_curriculum_epoch = getattr(
            config.training, 'ls_curriculum_epoch', 0)

        # SoftStage1 (默认 0)
        self.soft_stage1_epochs = getattr(
            config.training, 'soft_retriever_warmup_epochs', 0)
        self.soft_stage1_lr = getattr(
            config.training, 'soft_retriever_warmup_lr', 3e-4)
        self._in_soft_stage1 = False

        # 是否打印 per-query 检索诊断
        self.show_retrieval_diag = show_retrieval_diag

        # 同步到 loss_fn (保险兼容)
        if hasattr(self.loss_fn, 'set_label_smoothing'):
            self.loss_fn.set_label_smoothing(self.label_smoothing)
        elif hasattr(self.loss_fn, 'label_smoothing'):
            self.loss_fn.label_smoothing = self.label_smoothing

        self._setup_optimizer()

    def _restore_cuda_device(self):
        if self.device.type == 'cuda':
            idx = self.device.index if self.device.index is not None else 0
            try:
                torch.cuda.set_device(idx)
                _d = torch.zeros(1, device=f'cuda:{idx}'); del _d
            except Exception: pass

    def _setup_optimizer(self, base_lr=None, vis_proj_mult=3.0,
                         attr_mult=1.5, remaining_epochs=None):
        cfg = self.config.training
        lr = base_lr or cfg.stage2_lr
        if hasattr(self.model, 'get_trainable_param_groups'):
            param_groups = self.model.get_trainable_param_groups(
                base_lr=lr, vis_proj_mult=vis_proj_mult, attr_mult=attr_mult)
        else:
            param_groups = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(
            param_groups, lr=lr, weight_decay=cfg.weight_decay)
        steps_per_epoch = len(self.train_loader)
        n_epochs = remaining_epochs if remaining_epochs is not None else cfg.stage2_epochs
        total_steps = max(steps_per_epoch * n_epochs, 1)
        max_lrs = [g['lr'] if isinstance(g, dict) else lr
                   for g in self.optimizer.param_groups]
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer, max_lr=max_lrs, total_steps=total_steps,
            pct_start=max(cfg.warmup_ratio, 1.0/max(total_steps,1)),
            anneal_strategy='cos', div_factor=10.0, final_div_factor=100.0)
        lr_str = ', '.join(f'{l:.2e}' for l in max_lrs)
        self.logger.info(
            f"  [OneCycleLR] total_steps={total_steps} epochs={n_epochs} "
            f"steps/ep={steps_per_epoch} max_lr=[{lr_str}]")

    def _setup_soft_stage1_optimizer(self):
        stage1_params = [p for p in self.model.parameters() if p.requires_grad]
        if not stage1_params:
            self.logger.warning("  [SoftStage1] 无可训练参数，跳过"); return False
        self.optimizer = torch.optim.AdamW(
            stage1_params, lr=self.soft_stage1_lr,
            weight_decay=self.config.training.weight_decay)
        steps_per_epoch = len(self.train_loader)
        total_steps = max(steps_per_epoch * self.soft_stage1_epochs, 1)
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer, max_lr=[self.soft_stage1_lr],
            total_steps=total_steps,
            pct_start=max(0.1, 1.0/max(total_steps,1)),
            anneal_strategy='cos', div_factor=5.0, final_div_factor=50.0)
        n_total = sum(p.numel() for p in stage1_params)
        self.logger.info(
            f"  [SoftStage1 Opt] lr={self.soft_stage1_lr:.2e} "
            f"params={n_total/1e6:.2f}M total_steps={total_steps}")
        return True

    def _ensure_vision_loaded(self):
        """[v6.8] 日志反映 LAVIS eva_vit_g.pth + v6.8 attr_memory 同源"""
        if self.online_mode and not self._vision_loaded:
            self.logger.info("[ONLINE] 加载视觉编码器 (LAVIS EVA-ViT-g/14)...")
            self.model.load_vision_encoders()
            self._vision_loaded = True
            self._evaclip_init_done = True
            self.logger.info(
                "  [INIT-VIS] LAVIS eva_vit_g.pth 已加载 "
                "(与 ext_memory_lvis.pkl / attr_memory.pkl[v6.8] 构建路径一致) ✓")
            self.logger.info(
                "  [INIT-QF]  LAVIS Q-Former cross-attn K/V ← 预训练权重 ✓")

    def _extract_online_features(self, images, lvis_rois_batch=None):
        with torch.no_grad():
            return self.model.extract_online(images, lvis_boxes=lvis_rois_batch)

    def _get_retrieval_vecs(self, batch, n=None):
        patch = batch.get('eva_patches')
        if patch is None: return None
        if n is not None: patch = patch[:n]
        try:
            with torch.no_grad():
                _, retrieval_vecs = self.model.retrieval_qformer(patch.float())
                return retrieval_vecs
        except Exception:
            return None

    def _maybe_update_label_smoothing(self):
        """[LS-CUR] 默认关闭 (ls_curriculum_epoch=0)"""
        if self.ls_curriculum_epoch <= 0:
            return
        new_ls = 0.1 if self.epoch > self.ls_curriculum_epoch else self.label_smoothing
        current = getattr(self.loss_fn, 'label_smoothing', self.label_smoothing)
        if abs(new_ls - current) > 1e-6:
            if hasattr(self.loss_fn, 'set_label_smoothing'):
                self.loss_fn.set_label_smoothing(new_ls)
            elif hasattr(self.loss_fn, 'label_smoothing'):
                self.loss_fn.label_smoothing = new_ls
            self.label_smoothing = new_ls
            self.logger.info(
                f"  [LS-CUR] epoch={self.epoch} "
                f"label_smoothing {current:.2f} → {new_ls:.2f}")

    def _compute_loss(self, batch, logits, fwd_tuple=None):
        """仅 NTP cross-entropy (标准 ITG loss)"""
        labels = batch['text_tokens'][:, 1:].contiguous()
        if self.tokenizer:
            labels = labels.masked_fill(labels == self.tokenizer.pad_token_id, -100)
        L_txt = labels.shape[1]
        logits_t = logits[:, -L_txt:, :].contiguous()
        ntp_loss = F.cross_entropy(
            logits_t.view(-1, logits_t.size(-1)),
            labels.view(-1),
            ignore_index=-100,
            label_smoothing=self.label_smoothing,
        )
        total = self.w_ntp * ntp_loss
        details = {
            'NTPLoss':   ntp_loss.item(),
            'Contrast':  0.0,
            'RetAlign':  0.0,
            'NameAlign': 0.0,
        }
        return total, details

    def _get_extra_feats(self, batch, n=None):
        def _slice(key):
            v = batch.get(key)
            if v is None: return None
            return v[:n] if n is not None else v
        return {
            'eva_mid_patches': (_slice('eva_mid_patches') if self.use_ms_eva else None),
            'eva_global_per_roi': _slice('eva_global_per_roi'),
            'eva_global_per_roi_alt':    _slice('eva_global_per_roi_alt'),
            'eva_patches':              _slice('eva_patches'),
        }

    def _log_roi_coverage(self, batch, step):
        try:
            cat_ids = batch.get('category_ids', [])
            tag_seqs = batch.get('tag_sequences', [])
            total_roi = 0; covered = 0
            _invalid = {"object","scene","region","unknown","background","stuff",""}
            _coco_ids = set(range(1, 91))
            for b_idx in range(len(cat_ids)):
                cats_b = cat_ids[b_idx] if b_idx < len(cat_ids) else []
                tags_b = tag_seqs[b_idx] if b_idx < len(tag_seqs) else []
                n_roi = max(len(cats_b), len(tags_b))
                for i in range(n_roi):
                    total_roi += 1
                    cid = cats_b[i] if i < len(cats_b) else 0
                    tag = tags_b[i].strip().lower() if i < len(tags_b) else ""
                    if int(cid) in _coco_ids or (tag and tag not in _invalid):
                        covered += 1
            if total_roi > 0:
                rate = covered / total_roi
                msg = (f"  [ROI-COV] step={step} gstep={self.global_step} "
                       f"覆盖={covered}/{total_roi} ({rate:.1%})")
                if rate < 0.4: msg += "  ⚠️ <40%"
                self.logger.info(msg)
        except Exception:
            pass

    def train_epoch(self, desc_suffix=""):
        self.model.train()
        self._ensure_vision_loaded()
        self._maybe_update_label_smoothing()
        ds = self.train_loader.dataset
        if hasattr(ds, 'current_epoch'): ds.current_epoch = self.epoch
        total_loss = 0; n = 0
        use_sp = self.config.ablation.use_spatial
        use_at = self.config.ablation.use_attribute
        desc = f"Epoch {self.epoch}{desc_suffix}"
        pbar = tqdm(self.train_loader, desc=desc)

        for raw_batch in pbar:
            if self.online_mode and 'images' in raw_batch:
                images = raw_batch['images']
                lvis_rois = raw_batch.get('lvis_rois')
                vis_feats = self._extract_online_features(
                    images, lvis_rois_batch=lvis_rois)
                self._restore_cuda_device()
                batch = dict(vis_feats)
                batch['text_tokens']  = raw_batch['text_tokens'].to(self.device)
                batch['text_mask']    = raw_batch['text_mask'].to(self.device)
                batch['img_ids']      = raw_batch.get('img_ids', ['?']*len(images))
                batch['all_captions'] = raw_batch.get('all_captions', [])
            else:
                batch = {k: v.to(self.device) if torch.is_tensor(v) else v
                         for k, v in raw_batch.items()}

            self._restore_cuda_device()
            ex = self._get_extra_feats(batch)

            with _safe_autocast():
                fwd = self.model(
                    batch['eva_global'], batch['eva_global_768'],
                    batch['eva_rois'], batch['text_tokens'],
                    batch.get('text_mask'),
                    abs_pos=batch.get('absolute_pos'),
                    rel_pos=(batch.get('relative_pos') if use_sp else None),
                    attr_feats=(batch.get('attribute_features') if use_at else None),
                    tag_sequences=batch.get('tag_sequences'),
                    is_core=batch.get('is_core'),
                    importance_scores=batch.get('importance_scores'),
                    roi_mask=(batch.get('roi_mask') if use_sp else None),
                    category_ids=batch.get('category_ids'),
                    use_qformer=True, use_spatial_anchor=use_sp,
                    return_intermediates=True,
                    eva_patches=batch.get('eva_patches'),
                    eva_mid_patches=ex['eva_mid_patches'],
                    tag_confidence=batch.get('tag_confidence'),
                    eva_global_per_roi=ex['eva_global_per_roi'],
                    eva_global_per_roi_alt=ex['eva_global_per_roi_alt'],
                    boxes_batch=batch.get('boxes_batch'),
                )
                logits = fwd[0] if isinstance(fwd, tuple) else fwd
                loss, details = self._compute_loss(batch, logits, fwd_tuple=fwd)

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(),
                                      self.config.training.gradient_clip)
            self.optimizer.step()
            self.scheduler.step()

            self.global_step += 1
            total_loss += loss.item(); n += 1
            ntp = details.get('NTPLoss', loss.item())
            pbar.set_postfix_str(f"loss={loss.item():.4f} ntp={ntp:.4f}")

            if n % 20 == 0:
                ls_val = self.label_smoothing
                self.logger.info(
                    f"Ep{self.epoch}{desc_suffix} step={n} gstep={self.global_step} "
                    f"loss={loss.item():.4f} NTP={ntp:.4f} LS={ls_val:.2f}")

            if n % 100 == 0 and not (n % self.print_samples_every == 0 and self.tokenizer):
                self._log_roi_coverage(batch, n)

            if n % self.print_samples_every == 0 and self.tokenizer:
                self._show_inline_samples(batch, use_sp, use_at, n, ntp, 0.0, pbar=pbar)

        return total_loss / max(n, 1)

    def _show_inline_samples(self, batch, use_sp, use_at, step, ntp, ret_aln, pbar=None):
        n_show = min(3, batch['text_tokens'].shape[0])
        self.model.eval()
        ex = self._get_extra_feats(batch, n=n_show)
        retrieval_vecs_display = self._get_retrieval_vecs(batch, n=n_show)

        raw_names_display = None; attrs_display = None
        try:
            raw_names_display, attrs_display = (
                self.model._collect_retrieval_names_and_attrs(
                    n_show,
                    category_ids=batch.get('category_ids', [[]])[:n_show],
                    tag_sequences=(batch['tag_sequences'][:n_show]
                                   if 'tag_sequences' in batch else None),
                    is_core=batch.get('is_core', torch.zeros(1))[:n_show],
                    eva_global_768=batch['eva_global_768'][:n_show],
                    eva_rois=batch['eva_rois'][:n_show],
                    tag_confidence=batch.get('tag_confidence'),
                    eva_global_per_roi=ex['eva_global_per_roi'],
                    eva_global_per_roi_alt=ex['eva_global_per_roi_alt'],
                    retrieval_vecs=retrieval_vecs_display,
                ))
        except Exception as e:
            tqdm.write(f"  [DEBUG] _collect failed: {e}")

        try:
            with torch.no_grad():
                self._restore_cuda_device()
                with _safe_autocast():
                    pred_ids = self.model.generate(
                        batch['eva_global'][:n_show],
                        batch['eva_global_768'][:n_show],
                        batch['eva_rois'][:n_show],
                        abs_pos=batch.get('absolute_pos', torch.zeros(1))[:n_show],
                        rel_pos=(batch.get('relative_pos', torch.zeros(1))[:n_show]
                                 if use_sp else None),
                        attr_feats=(batch.get('attribute_features', torch.zeros(1))[:n_show]
                                    if use_at else None),
                        tag_sequences=(batch['tag_sequences'][:n_show]
                                       if 'tag_sequences' in batch else None),
                        is_core=batch.get('is_core', torch.zeros(1))[:n_show],
                        importance_scores=batch.get('importance_scores', torch.zeros(1))[:n_show],
                        category_ids=batch.get('category_ids', [[]])[:n_show],
                        use_qformer=True, use_spatial_anchor=use_sp,
                        eva_patches=(batch.get('eva_patches')[:n_show]
                                      if batch.get('eva_patches') is not None else None),
                        eva_mid_patches=ex['eva_mid_patches'],
                        tag_confidence=batch.get('tag_confidence'),
                        eva_global_per_roi=ex['eva_global_per_roi'],
                        eva_global_per_roi_alt=ex['eva_global_per_roi_alt'],
                        boxes_batch=batch.get('boxes_batch', None),
                        **_SAMPLE_GEN_KWARGS)
        finally:
            self.model.train()

        pad_id = self.tokenizer.pad_token_id
        impl_names_all = getattr(self.model, '_last_implicit_names', None)
        vote_topk_all = getattr(self.model, '_last_vote_topk', None)
        vote_denoised_all = getattr(self.model, '_last_vote_denoised', None)
        all_captions_batch = batch.get('all_captions', [])

        token_layout = _get_token_layout(self.model)
        ls_val = self.label_smoothing
        vis_str = "LAVIS-ViT✓" if self._evaclip_init_done else "LAVIS-ViT○"

        lines = [
            f"─── step={step} ×{n_show} [NTP={ntp:.4f} LS={ls_val:.2f}] "
            f"Token={token_layout} {vis_str} [5GT] ───"
        ]

        for i in range(n_show):
            tgt = batch['text_tokens'][i]
            gt_selected = self.tokenizer.decode(
                tgt[tgt != pad_id].tolist(), skip_special_tokens=True).strip()
            pred = self.model.decode_and_clean(pred_ids[i], self.tokenizer)

            tag_str = ""
            tags = batch.get('tag_sequences')
            if tags and i < len(tags):
                real_tags = _filter_real_tags(tags[i], max_show=5)
                if real_tags: tag_str = f" [{','.join(real_tags)}]"

            inject_str = ""; attr_detail_str = ""
            if raw_names_display is not None and i < len(raw_names_display):
                names_i = raw_names_display[i]
                attrs_i = (attrs_display[i] if attrs_display and i < len(attrs_display) else {})
                fmt_str = _fmt_name_attrs(names_i, attrs_i, max_names=4)
                if fmt_str: inject_str = f" → PKL:[{fmt_str[:100]}]"
                attr_fmt = _fmt_attr_details(names_i, attrs_i, max_names=4)
                if attr_fmt: attr_detail_str = f"\n       属性:   {attr_fmt}"

            retrieval_str = ""
            topk_b = (vote_topk_all[i] if vote_topk_all and i < len(vote_topk_all) else [])
            dn_b = (vote_denoised_all[i] if vote_denoised_all and i < len(vote_denoised_all) else [])
            gt_caps_i = (all_captions_batch[i] if i < len(all_captions_batch) else gt_selected)
            if topk_b or dn_b:
                retrieval_str = "\n" + _format_retrieval_diag(
                    topk_b, dn_b, gt_caps_i, max_show=6)

            # per-query 检索诊断
            evcap_retr_str = ""
            if self.show_retrieval_diag and hasattr(self.model, 'format_evcap_retrieval_diag'):
                try:
                    diag = self.model.format_evcap_retrieval_diag(b=i, show_queries=8)
                    if diag: evcap_retr_str = "\n" + diag
                except Exception:
                    pass

            impl_str = ""
            if impl_names_all is not None and i < len(impl_names_all):
                valid_names = [nm for nm in impl_names_all[i] if nm]
                if valid_names:
                    impl_str = f" soft:[{', '.join(valid_names[:4])}]"

            lines.append("")
            lines.append(f"  [{i+1}]{tag_str}{inject_str}{impl_str}"
                         f"{attr_detail_str}{retrieval_str}{evcap_retr_str}")
            gt_all_i = all_captions_batch[i] if i < len(all_captions_batch) else []
            if not gt_all_i: gt_all_i = [gt_selected]
            elif isinstance(gt_all_i, str): gt_all_i = [gt_selected]
            for gi, gc in enumerate(gt_all_i):
                marker = " ◀" if gc.strip() == gt_selected.strip() else ""
                lines.append(f"       GT[{gi+1}]: {gc}{marker}")
            lines.append(f"       预测: {pred if pred else '[空]'}")

        output = "\n".join(lines)
        if pbar is not None: pbar.write(output)
        else: tqdm.write(output)
        self.logger.info(
            f"  [SAMPLE] step={step} ×{n_show} NTP={ntp:.4f} (详细预测见终端)")

    @torch.no_grad()
    def show_samples(self, n_samples=3):
        if not self.tokenizer: return
        self.model.eval()
        try:
            raw_batch = next(iter(self.train_loader))
            if self.online_mode and 'images' in raw_batch:
                self._ensure_vision_loaded()
                vis_feats = self._extract_online_features(
                    raw_batch['images'], raw_batch.get('lvis_rois'))
                batch = dict(vis_feats)
                batch['text_tokens']  = raw_batch['text_tokens'].to(self.device)
                batch['text_mask']    = raw_batch['text_mask'].to(self.device)
                batch['all_captions'] = raw_batch.get('all_captions', [])
            else:
                batch = {k: v.to(self.device) if torch.is_tensor(v) else v
                         for k, v in raw_batch.items()}
            n_show = min(n_samples, batch['text_tokens'].shape[0])
            use_sp = self.config.ablation.use_spatial
            ex = self._get_extra_feats(batch, n=n_show)
            retrieval_vecs_display = self._get_retrieval_vecs(batch, n=n_show)

            raw_names_display = None; attrs_display = None
            try:
                raw_names_display, attrs_display = (
                    self.model._collect_retrieval_names_and_attrs(
                        n_show,
                        category_ids=batch.get('category_ids', [[]])[:n_show],
                        tag_sequences=(batch['tag_sequences'][:n_show]
                                       if 'tag_sequences' in batch else None),
                        is_core=batch.get('is_core', torch.zeros(1))[:n_show],
                        eva_global_768=batch['eva_global_768'][:n_show],
                        eva_rois=batch['eva_rois'][:n_show],
                        tag_confidence=batch.get('tag_confidence'),
                        eva_global_per_roi=ex['eva_global_per_roi'],
                        eva_global_per_roi_alt=ex['eva_global_per_roi_alt'],
                        retrieval_vecs=retrieval_vecs_display,
                    ))
            except Exception:
                pass

            self._restore_cuda_device()
            with _safe_autocast():
                pred_ids = self.model.generate(
                    batch['eva_global'][:n_show],
                    batch['eva_global_768'][:n_show],
                    batch['eva_rois'][:n_show],
                    abs_pos=batch.get('absolute_pos', torch.zeros(1))[:n_show],
                    rel_pos=(batch.get('relative_pos', torch.zeros(1))[:n_show]
                             if use_sp else None),
                    is_core=batch.get('is_core', torch.zeros(1))[:n_show],
                    category_ids=batch.get('category_ids', [[]])[:n_show],
                    tag_sequences=(batch['tag_sequences'][:n_show]
                                   if 'tag_sequences' in batch else None),
                    use_qformer=True, use_spatial_anchor=use_sp,
                    eva_patches=(batch.get('eva_patches')[:n_show]
                                  if batch.get('eva_patches') is not None else None),
                    eva_mid_patches=ex['eva_mid_patches'],
                    tag_confidence=batch.get('tag_confidence'),
                    eva_global_per_roi=ex['eva_global_per_roi'],
                    eva_global_per_roi_alt=ex['eva_global_per_roi_alt'],
                    boxes_batch=batch.get('boxes_batch', None),
                    **_EVAL_GEN_KWARGS)

            pad_id = self.tokenizer.pad_token_id
            trainable_m = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            token_layout = _get_token_layout(self.model)
            ls_val = self.label_smoothing
            vis_str = "LAVIS-ViT✓" if self._evaclip_init_done else "LAVIS-ViT未触发"

            lines = [
                "=" * 72,
                f"样本预测×{n_show}  Epoch {self.epoch}  "
                f"model v6.6.1 / train v6.2.1 / attr_memory v6.8",
                f"  可训练: {trainable_m/1e6:.2f}M | beam=5 | "
                f"max_new=60 | LS={ls_val:.2f} | [5GT展开]",
                f"  Token: {token_layout} | {vis_str}",
                f"  [v6.6.1] 视觉: LAVIS eva_vit_g.pth (与 ext_memory_lvis.pkl 同源)",
                f"  [v6.8]   attr_memory: VAW + VG attribute_types 合并 (~94 dim)",
                f"  [PER-Q]  检索: per-query-token (32个query独立top-5→合并去重)",
                f"  [QF]     Q-Former (双FFN, 跳过 position_embed)",
                f"  [PKL]    pre_name 清洗 (同源 768 维检索空间)",
                f"  [LOSS]   仅 NTP cross-entropy, label_smoothing={ls_val:.2f}",
                "=" * 72,
            ]

            impl_names_all = getattr(self.model, '_last_implicit_names', None)
            vote_topk_all = getattr(self.model, '_last_vote_topk', None)
            vote_denoised_all = getattr(self.model, '_last_vote_denoised', None)
            all_captions_batch = batch.get('all_captions', [])

            for i in range(n_show):
                tgt = batch['text_tokens'][i]
                gt_selected = self.tokenizer.decode(
                    tgt[tgt != pad_id].tolist(), skip_special_tokens=True).strip()
                pred = self.model.decode_and_clean(pred_ids[i], self.tokenizer)

                tag_str = ""
                tags = batch.get('tag_sequences')
                if tags and i < len(tags):
                    real_tags = _filter_real_tags(tags[i], max_show=5)
                    if real_tags: tag_str = f" [{','.join(real_tags)}]"

                inject_str = ""; attr_detail_str = ""
                if raw_names_display is not None and i < len(raw_names_display):
                    names_i = raw_names_display[i]
                    attrs_i = (attrs_display[i] if attrs_display and i < len(attrs_display) else {})
                    fmt_str = _fmt_name_attrs(names_i, attrs_i, max_names=4)
                    if fmt_str: inject_str = f" → PKL:[{fmt_str[:100]}]"
                    attr_fmt = _fmt_attr_details(names_i, attrs_i, max_names=4)
                    if attr_fmt: attr_detail_str = f"\n    属性:   {attr_fmt}"

                retrieval_str = ""
                topk_b = (vote_topk_all[i] if vote_topk_all and i < len(vote_topk_all) else [])
                dn_b = (vote_denoised_all[i] if vote_denoised_all and i < len(vote_denoised_all) else [])
                gt_caps_i = (all_captions_batch[i] if i < len(all_captions_batch) else gt_selected)
                if topk_b or dn_b:
                    retrieval_str = "\n" + _format_retrieval_diag(
                        topk_b, dn_b, gt_caps_i, max_show=8)

                evcap_retr_str = ""
                if self.show_retrieval_diag and hasattr(self.model, 'format_evcap_retrieval_diag'):
                    try:
                        diag = self.model.format_evcap_retrieval_diag(b=i, show_queries=8)
                        if diag: evcap_retr_str = "\n" + diag
                    except Exception:
                        pass

                impl_str = ""
                if impl_names_all is not None and i < len(impl_names_all):
                    valid_names = [nm for nm in impl_names_all[i] if nm]
                    if valid_names:
                        impl_str = f" soft:[{', '.join(valid_names[:4])}]"

                lines.append(
                    f"  GT:{tag_str}{inject_str}{impl_str}"
                    f"{attr_detail_str}{retrieval_str}{evcap_retr_str}")
                gt_all_i = all_captions_batch[i] if i < len(all_captions_batch) else []
                if not gt_all_i: gt_all_i = [gt_selected]
                elif isinstance(gt_all_i, str): gt_all_i = [gt_selected]
                for gi, gc in enumerate(gt_all_i):
                    marker = " ◀" if gc.strip() == gt_selected.strip() else ""
                    lines.append(f"    GT[{gi+1}]: {gc}{marker}")
                lines.append(f"  预测: {pred if pred else '[空]'}")

            tqdm.write("\n".join(lines))
            self.logger.info(
                f"  [SAMPLE] epoch={self.epoch} ×{n_show} 预测样本 (详细见终端)")
        finally:
            self.model.train()

    def save(self, path):
        state = {k: v for k, v in self.model.state_dict().items()
                 if 'decoder.llm' not in k or 'lora_' in k}
        torch.save({'model': state, 'stage': self.stage,
                    'epoch': self.epoch, 'global_step': self.global_step}, path)
        self.logger.info(f"已保存: {path}")

    def _get_ckpt_dir(self):
        cfg = self.config.training
        ckpt_dir = (getattr(cfg, 'checkpoint_dir', None)
                    or getattr(self.config.paths, 'checkpoint_dir', None)
                    or getattr(self.config.paths, 'save_dir', None)
                    or getattr(self.config.paths, 'output_dir', None)
                    or '.')
        os.makedirs(ckpt_dir, exist_ok=True)
        return ckpt_dir

    def train(self):
        cfg = self.config.training
        self.logger.info("=" * 72)
        self.logger.info(
            "V6.2.1 训练 ← model v6.6.1 + attr_memory v6.8 | "
            "LAVIS 视觉对齐 + per-query 检索 + Karpathy 118k")
        self.logger.info("  [v6.6.1-VIS] 视觉: LAVIS eva_vit_g.pth (与 ext_memory_lvis.pkl 同源)")
        self.logger.info("  [v6.8-ATTR]  attr_memory: VAW + VG attribute_types 合并 (~94 dim)")
        self.logger.info("  [PER-Q]      per-query-token 检索 (32q 独立top-5 合并去重)")
        self.logger.info("  [QF]         Q-Former (双FFN, 跳过 position_embed)")
        self.logger.info("  [PKL]        同源 768 维检索空间 (pre_name 清洗)")
        self.logger.info("  [LOSS]       仅 NTP cross-entropy, "
                         f"label_smoothing={self.label_smoothing:.2f}")
        self.logger.info("  [RETR]       诊断: per-query top-1 打印已启用")
        self.logger.info("  [5GT]        5 GT 展开训练")
        self.logger.info(f"  [LS-CUR]     ls_curriculum_epoch={self.ls_curriculum_epoch} "
                         f"(0=关闭)")
        self.logger.info(f"  [SOFT]       warmup_epochs={self.soft_stage1_epochs} "
                         f"(默认0, 已移除 soft_retriever 训练)")
        self.logger.info("=" * 72)

        if (self.soft_stage1_epochs > 0
                and hasattr(self.model, 'soft_retriever')
                and hasattr(self.model, 'freeze_for_soft_stage1')):
            self.logger.info("=" * 72)
            n_streams = len(getattr(self.model.soft_retriever, 'streams', []))
            self.logger.info(
                f"=== SoftStage1: soft_retriever({n_streams}流) "
                f"({self.soft_stage1_epochs}ep, lr={self.soft_stage1_lr:.2e}) ===")
            self.logger.info("=" * 72)
            self.model.freeze_for_soft_stage1()
            self._in_soft_stage1 = True
            ok = self._setup_soft_stage1_optimizer()
            if ok:
                ckpt_dir = self._get_ckpt_dir()
                for ep in range(1, self.soft_stage1_epochs + 1):
                    self.epoch = ep
                    train_loss = self.train_epoch(desc_suffix=" [SoftStage1]")
                    n_soft = sum(p.numel() for p in self.model.parameters()
                                 if p.requires_grad)
                    self.logger.info(
                        f"SoftStage1 Ep{ep}/{self.soft_stage1_epochs} "
                        f"Loss={train_loss:.4f} | soft_params={n_soft/1e6:.2f}M | "
                        f"gstep={self.global_step}")
                    self.show_samples()
                    self.save(os.path.join(ckpt_dir, f'soft_stage1_ep{ep}.pt'))
                self.logger.info("SoftStage1 完成!")
            else:
                self.logger.warning("SoftStage1 跳过 (无可训练参数)")
            self._in_soft_stage1 = False

        # Stage2
        self.logger.info("=" * 72)
        self.logger.info(
            f"=== Stage2 ({cfg.stage2_epochs}轮 NTP, lr={cfg.stage2_lr}) ===")
        self.logger.info("  调用 model.unfreeze_for_stage2()")
        self.logger.info("=" * 72)
        self.stage = 2
        if hasattr(self.model, 'unfreeze_for_stage2'):
            self.model.unfreeze_for_stage2()
        if (hasattr(self.model, 'decoder')
                and hasattr(self.model.decoder, 'set_prompt_stage')):
            self.model.decoder.set_prompt_stage("stage2_early")

        start_ep = self.epoch + 1 if self.epoch > 0 else 1
        total_ep = cfg.stage2_epochs
        ckpt_dir = self._get_ckpt_dir()
        remaining_ep = total_ep - start_ep + 1
        if remaining_ep <= 0:
            self.logger.info(
                f"Stage2 已完成全部 {total_ep} 轮 (当前 epoch={self.epoch})")
            self.save(os.path.join(ckpt_dir, 'final.pt'))
            self.logger.info("训练完成!")
            return

        self.logger.info(
            f"  [续训] start_ep={start_ep} total_ep={total_ep} "
            f"remaining={remaining_ep} gstep={self.global_step}")
        self._setup_optimizer(base_lr=cfg.stage2_lr,
                              vis_proj_mult=3.0, attr_mult=1.5,
                              remaining_epochs=remaining_ep)

        for ep in range(start_ep, total_ep + 1):
            self.epoch = ep
            if (ep == self.unfreeze_new_modules_epoch
                    and hasattr(self.model, 'unfreeze_new_modules')):
                self.logger.info(f"  [epoch{ep}] unfreeze_new_modules() 触发")
                total_m = self.model.unfreeze_new_modules()
                remaining_after = total_ep - ep + 1
                self._setup_optimizer(
                    base_lr=cfg.stage2_lr,
                    vis_proj_mult=3.0, attr_mult=1.5,
                    remaining_epochs=remaining_after)
                self.logger.info(
                    f"  [epoch{ep}] scheduler 重建 remaining={remaining_after}ep | "
                    f"总可训练: {total_m/1e6:.2f}M")

            train_loss = self.train_epoch(desc_suffix=" [Stage2]")
            trainable_m = sum(p.numel() for p in self.model.parameters()
                              if p.requires_grad)
            ls_val = self.label_smoothing
            vis_status = "✓ LAVIS-ViT 初始化" if self._evaclip_init_done else "○ 未初始化"
            self.logger.info(
                f"Ep {ep}/{total_ep} [S2-NTP] Loss={train_loss:.4f} | "
                f"trainable={trainable_m/1e6:.2f}M | LS={ls_val:.2f} | "
                f"{vis_status} | gstep={self.global_step}")
            if (ep == total_ep
                    and hasattr(self.model.decoder, 'set_prompt_stage')):
                self.model.decoder.set_prompt_stage("stage2_late")
            self.show_samples()
            self.save(os.path.join(ckpt_dir, f'stage2_ep{ep}.pt'))

        self.save(os.path.join(ckpt_dir, 'stage2_final.pt'))
        self.logger.info("Stage2 NTP 完成!")
        self.save(os.path.join(ckpt_dir, 'final.pt'))
        self.logger.info("训练完成!")


# ─────────────────────────────────────────────────────────────
# main  [V6.2.1: --train_json CLI 显式接收 Karpathy 路径]
# ─────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="V6.2.1 训练 (model v6.6.1 + attr_memory v6.8 / "
                    "LAVIS 视觉对齐 / Karpathy 118k 适配)")
    parser.add_argument('--ablation',            default='full')
    parser.add_argument('--image_dir',           default=None)
    parser.add_argument('--caption_file',        default=None)
    parser.add_argument('--train_json',          default=None,
                        help='★ V6.2.1: Karpathy 训练 JSON 路径 (优先级最高, '
                             '覆盖 --caption_file 和 config 默认值)')
    parser.add_argument('--resume',              default=None)
    parser.add_argument('--blip2_qformer_path',  default=None)
    parser.add_argument('--num_captions',        type=int, default=5)
    parser.add_argument('--lvis_json',           default=None)
    parser.add_argument('--lvis_val_json',       default=None,
                        help='★ V6.2.1: val instances JSON (Karpathy 必需)')
    parser.add_argument('--soft_warmup_epochs',  type=int, default=None)
    parser.add_argument('--yoloe_path',          default=None)
    parser.add_argument('--yoloe_conf',          type=float, default=0.25)
    parser.add_argument('--ls_curriculum_epoch', type=int, default=None,
                        help='前N epoch 低 LS，之后 LS=0.1 (0=不启用)')
    parser.add_argument('--label_smoothing', type=float, default=0.0,
                        help='Label smoothing (默认=0.0)')
    parser.add_argument('--show_retrieval_diag', action='store_true', default=True,
                        help='打印 per-query 检索诊断 (32个query每个的top-1)')
    parser.add_argument('--no_retrieval_diag', action='store_false',
                        dest='show_retrieval_diag',
                        help='关闭 per-query 检索诊断')
    args = parser.parse_args()
    args.num_captions = 5  # 强制 5 GT

    config = get_ablation_config(args.ablation)
    create_directories(config)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(
                f"{config.paths.log_dir}/{config.experiment_name}.log"),
            logging.StreamHandler(),
        ])
    logger = logging.getLogger()
    print_config(config)

    if args.soft_warmup_epochs is not None:
        config.training.soft_retriever_warmup_epochs = args.soft_warmup_epochs
        logger.info(f"  [CLI] soft_warmup_epochs={args.soft_warmup_epochs}")
    if args.ls_curriculum_epoch is not None:
        config.training.ls_curriculum_epoch = args.ls_curriculum_epoch
        logger.info(f"  [CLI] ls_curriculum_epoch={args.ls_curriculum_epoch}")

    logger.info(f"  [LOSS] label_smoothing={args.label_smoothing:.2f}")
    logger.info(f"  [RETR] show_retrieval_diag={args.show_retrieval_diag}")

    # 模型初始化
    model = CaptionModel(config)
    total_m = sum(p.numel() for p in model.parameters())
    logger.info(f"总参数: {total_m/1e6:.2f}M")

    if model._pkl_retriever is not None and model._pkl_retriever._loaded:
        r = model._pkl_retriever
        logger.info(f"  [PKL] ✅ 名称库: {r.name_keys.shape}  {len(r._name2idx)}类")
        if r.attr_keys is not None:
            all_dims = r.get_all_attr_dims()
            logger.info(f"  [PKL] ✅ 属性库: {r.attr_keys.shape}  "
                        f"维度({len(all_dims)})")
            _log_attr_memory_summary(
                getattr(config.paths, 'attr_pkl', None), logger)
        logger.info(f"  [PKL] 阈值: base={r.name_sim_threshold:.2f}")
    else:
        logger.warning("  [PKL] ⚠️ 未就绪")

    logger.info("  [INIT-QF] Q-Former 在 load_blip2_pretrained() 后加载")

    blip2_path = (args.blip2_qformer_path
                  or getattr(config.paths, 'blip2_qformer_path', None))
    if blip2_path and os.path.exists(blip2_path):
        model.load_blip2_pretrained(blip2_path)
        logger.info("  [INIT] Q-Former 预训练权重加载完成 ✓")
    else:
        logger.warning(f"  [QF] ⚠️ 未找到 Q-Former 预训练权重: {blip2_path}")

    # 仅 NTP loss
    loss_fn = CaptionLoss(
        w1=1.0, w2=0.0, w3=0.0, w4=0.0,
        ignore_index=-100, label_smoothing=args.label_smoothing)

    # ★★★ [V6.2.1 NEW] 数据集路径解析 (--train_json 优先) ★★★
    # 优先级:
    #   1. --train_json (Karpathy 路径, run.sh 显式传)
    #   2. --caption_file (CLI 兼容旧用法)
    #   3. config.paths.coco_karpathy_train_json (yaml 里 V6.5 新加)
    #   4. config.paths.coco_train_captions (COCO 原版 fallback)
    # 格式由 OnlineCaptionDataset 自动识别 (list = Karpathy, dict = COCO)
    caption_file = (args.train_json
                    or args.caption_file
                    or getattr(config.paths, 'coco_karpathy_train_json', None)
                    or getattr(config.paths, 'coco_train_captions', None))
    image_dir = (args.image_dir
                 or getattr(config.paths, 'coco_train_images', None))
    train_inst_json = (args.lvis_json
                       or getattr(config.paths, 'lvis_train_json', None))
    val_inst_json   = (args.lvis_val_json
                       or getattr(config.paths, 'lvis_val_json', None))

    logger.info(f"")
    logger.info(f"  ━━━ [V6.2.1] 数据集路径解析 ━━━")
    logger.info(f"    image_dir:       {image_dir}")
    logger.info(f"    caption_file:    {caption_file}")
    logger.info(f"    train instances: {train_inst_json}")
    logger.info(f"    val instances:   {val_inst_json}  ★ V6.2.1 DUAL-INST")
    if args.train_json:
        logger.info(f"    [来源] --train_json CLI 参数 (Karpathy)")
    elif args.caption_file:
        logger.info(f"    [来源] --caption_file CLI 参数")
    elif getattr(config.paths, 'coco_karpathy_train_json', None):
        logger.info(f"    [来源] config.paths.coco_karpathy_train_json")
    else:
        logger.info(f"    [来源] config.paths.coco_train_captions (COCO 原版)")
    logger.info(f"")

    # CocoBox 加载 (★ V6.2.1: 双 instances 合并)
    coco_box_loader = None
    if train_inst_json and os.path.exists(train_inst_json):
        coco_box_loader = CocoBoxLoader(
            train_inst_json,
            val_json_path=val_inst_json,  # ★ V6.2.1: None 时跟 v6.0 一样, 非 None 时合并
            max_rois=config.training.max_rois,
            core_count=getattr(config.training, 'core_rois', 4))
        if not coco_box_loader.is_loaded:
            coco_box_loader = None

    # 数据集
    if not image_dir or not os.path.isdir(image_dir):
        logger.error(f"需要图片目录 (image_dir 不存在或无效): {image_dir}"); return
    if not caption_file or not os.path.exists(caption_file):
        logger.error(f"需要 caption 文件 (不存在): {caption_file}"); return

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.paths.llm_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'

    train_ds = OnlineCaptionDataset(
        image_dir=image_dir, caption_file=caption_file,
        tokenizer=tokenizer,
        max_len=getattr(config.training, 'max_len', 64),
        num_captions_per_image=args.num_captions,
        lvis_loader=coco_box_loader,
        yoloe_detector=None,
        max_rois=config.training.max_rois,
        core_count=getattr(config.training, 'core_rois', 4),
    )

    # ★★★ FAST IO patch (v6.4): 修 GPU-Util 8% 的瓶颈 ★★★
    _nw = getattr(config.training, 'dataloader_num_workers', 16)
    if _nw is None or _nw <= 0:
        _nw = 16
    _pf = getattr(config.training, 'prefetch_factor', 4)
    _pw = getattr(config.training, 'persistent_workers', True)
    _pm = getattr(config.training, 'pin_memory', True)
    logger.info(f"  [DATALOADER] num_workers={_nw} prefetch_factor={_pf} "
                f"persistent_workers={_pw} pin_memory={_pm}")
    train_loader = DataLoader(
        train_ds,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=_nw,
        pin_memory=_pm,
        persistent_workers=(_pw and _nw > 0),
        prefetch_factor=(_pf if _nw > 0 else None),
        collate_fn=online_collate_fn,
        drop_last=True,
    )

    steps_per_epoch = len(train_loader)
    logger.info(f"  [SPEED] batch_size={config.training.batch_size} "
                f"steps/epoch={steps_per_epoch} "
                f"total_steps={steps_per_epoch * config.training.stage2_epochs}")

    trainer = Trainer(
        config, model, loss_fn, train_loader, logger,
        online_mode=True,
        label_smoothing=args.label_smoothing,
        show_retrieval_diag=args.show_retrieval_diag,
    )
    trainer.tokenizer = tokenizer
    model.set_tokenizer(tokenizer)

    if args.resume:
        ckpt = torch.load(args.resume, map_location='cpu', weights_only=False)
        missing, unexpected = trainer.model.load_state_dict(
            ckpt['model'], strict=False)
        logger.info(f"续传: 缺失={len(missing)} 多余={len(unexpected)}")
        _classify_resume_mismatch(missing, unexpected, logger)
        if missing: logger.info(f"  缺失示例(前5): {missing[:5]}")
        if unexpected: logger.info(f"  多余示例(前5): {unexpected[:5]}")
        trainer.stage = 2
        trainer.epoch = ckpt.get('epoch', 0)
        trainer.global_step = ckpt.get('global_step', 0)
        logger.info(f"从 epoch={trainer.epoch} gstep={trainer.global_step} 恢复")
        if trainer.epoch > 0:
            trainer.soft_stage1_epochs = 0
            logger.info("  续训: 跳过 SoftStage1")

    trainer.train()


if __name__ == "__main__":
    main()