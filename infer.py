#!/usr/bin/env python3
"""
infer_e2e_yoloe.py  v6.2.1-YOLOE
================
基于 infer_e2e.py v6.2.1, 唯一改动: ROI 来源从 GT instances_val2014.json
改为 YOLOe 在线检测 (COCO 80 类), 实现完全 free / end-to-end setup.

适配: caption_model 包 (v6.4 拆分, v6.6.1 model + v6.9.3 retriever)
      + train.py v6.2.1 (Karpathy 118k) + Karpathy test split

v6.2.1-YOLOE 改动 (相对 v6.2.1):
  [YOLOE-ROI]   ★ CocoValBoxLoader → YOLOeBoxLoader
                   ROI 来源: YOLOe > grid (完全不读 instances_*.json)
                   Free / end-to-end setup, 无 GT 泄露
  [YOLOE-80]    ★ COCO 80 类限定 + 同义词修正 (sofa→couch 等)
  [YOLOE-CLI]   ★ --val_box_json/--no_val_roi → --yoloe_path/--yoloe_conf/--no_yoloe

继承自 v6.2.1 (完全不变):
  [V6.2.1-IMPORT]  from caption_model import ... (v6.4 包结构)
  [V6.2.1-CKPT]    _DEFAULT_CKPT_PATH = checkpoints/stage2_ep1.pt
  [LAVIS-VIS]   视觉编码器: LAVIS eva_vit_g.pth (与 ext_memory_lvis.pkl 同源)
  [INIT-LAVIS]  Q-Former cross-attn K/V ← LAVIS eva_vit_g.pth
  [PER-Q-RETR]  per-query-token 检索 (32 个 query 独立 top-5 合并去重)
  [DYN-TOKEN]   Token 布局: 动态 "32vis + N流" (从 soft_retriever.streams 读取)
  [GEN-KWARGS]  与 train.py _EVAL_GEN_KWARGS 严格对齐:
                beam=5 / max_new=60 / min_new=8 / no_repeat_ngram=3 / rep=1.0
  CocoValBoxLoader→YOLOeBoxLoader, 单通道投票诊断, PKL 三步检索,
  run_eval/single/batch
"""

import os
import sys
import json
import glob
import re
import time
import argparse
import tempfile
from typing import Dict, List, Optional, Tuple
from collections import defaultdict, Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from tqdm import tqdm

# ── CUDA / autocast 初始化 ─────────────────────────────────────
if torch.cuda.is_available():
    try:
        _BF16_OK = torch.cuda.is_bf16_supported()
        _dummy = torch.zeros(1, device='cuda:0'); del _dummy
    except Exception:
        _BF16_OK = False
else:
    _BF16_OK = False


def _safe_autocast():
    if _BF16_OK:
        return torch.amp.autocast('cuda', dtype=torch.bfloat16)
    return torch.cuda.amp.autocast(dtype=torch.float16)


from config import get_config, get_ablation_config

# ★★ V6.2.1: 从 caption_model 包导入 (适配 v6.4 包结构) ★★
# v6.4 包 __init__ 已经把 _clean_caption / _COCO_ID_TO_NAME 映射到包内, 完全兼容
from caption_model import CaptionModel, _clean_caption

try:
    from caption_model import _COCO_ID_TO_NAME
except ImportError:
    # 兜底: 包里没导出就用本地表 (基本不会触发)
    _COCO_ID_TO_NAME = {
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


# ================================================================
# [YOLOE-80] COCO 80 类名 + name→id 映射 + 同义词
# ================================================================

# 顺序: 按 COCO category_id 排列 (跳过 12/26/29/30/45/66/68/69/71/83)
COCO_80_NAMES = [
    "person", "bicycle", "car", "motorcycle", "airplane",
    "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat",
    "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon",
    "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut",
    "cake", "chair", "couch", "potted plant", "bed",
    "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven",
    "toaster", "sink", "refrigerator", "book", "clock",
    "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
]

# 从 _COCO_ID_TO_NAME 反转构建 name→id 映射
_COCO_NAME_TO_ID = {v: k for k, v in _COCO_ID_TO_NAME.items()}

# YOLOe / ultralytics 可能输出的同义词 → COCO 标准名
_COCO_SYNONYMS = {
    "sofa": "couch",
    "motorbike": "motorcycle",
    "aeroplane": "airplane",
    "pottedplant": "potted plant",
    "potted_plant": "potted plant",
    "diningtable": "dining table",
    "dining_table": "dining table",
    "tvmonitor": "tv",
    "tv_monitor": "tv",
    "cellphone": "cell phone",
    "cell_phone": "cell phone",
    "hotdog": "hot dog",
    "hot_dog": "hot dog",
    "wineglass": "wine glass",
    "wine_glass": "wine glass",
    "baseballbat": "baseball bat",
    "baseball_bat": "baseball bat",
    "baseballglove": "baseball glove",
    "baseball_glove": "baseball glove",
    "tennisracket": "tennis racket",
    "tennis_racket": "tennis racket",
    "sportsball": "sports ball",
    "sports_ball": "sports ball",
    "trafficlight": "traffic light",
    "traffic_light": "traffic light",
    "firehydrant": "fire hydrant",
    "fire_hydrant": "fire hydrant",
    "stopsign": "stop sign",
    "stop_sign": "stop sign",
    "parkingmeter": "parking meter",
    "parking_meter": "parking meter",
    "teddybear": "teddy bear",
    "teddy_bear": "teddy bear",
    "hairdrier": "hair drier",
    "hair_drier": "hair drier",
    "hairdryer": "hair drier",
}


def _resolve_coco_id(name: str) -> int:
    """将 YOLOe 输出的类名解析为 COCO category_id (0=未知)。"""
    nm = name.strip().lower()
    # 直接匹配
    if nm in _COCO_NAME_TO_ID:
        return _COCO_NAME_TO_ID[nm]
    # 同义词
    canonical = _COCO_SYNONYMS.get(nm, None)
    if canonical and canonical in _COCO_NAME_TO_ID:
        return _COCO_NAME_TO_ID[canonical]
    return 0


# 推理侧统一阈值常量
_VOTE_THRESH = 5


# ================================================================
# 推理生成参数 — 与 train.py v6.2.1 _EVAL_GEN_KWARGS 严格对齐
# ================================================================
_EVAL_GEN_KWARGS = dict(
    max_new_tokens=60,
    min_new_tokens=8,
    num_beams=5,
    length_penalty=1.0,
    repetition_penalty=1.0,
    no_repeat_ngram_size=3,
    early_stopping=True,
)


# ================================================================
# [YOLOE-ROI] YOLOe 在线检测 Box 加载器 (替代 CocoValBoxLoader)
# ================================================================

class YOLOeBoxLoader:
    """
    使用 YOLOe 模型在线检测 ROI, 替代 GT instances_*.json.
    完全 free / end-to-end setup, 无 GT 泄露.
    """
    def __init__(self, model_path: str,
                 max_rois: int = 11, core_count: int = 4,
                 conf: float = 0.25, imgsz: int = 640):
        self.max_rois   = max_rois
        self.core_count = core_count
        self.conf       = conf
        self.imgsz      = imgsz
        self.is_loaded  = False
        self._model     = None
        self._model_names: Dict[int, str] = {}
        self._cache: Dict[str, list] = {}   # img_id → boxes 缓存

        if not model_path or not os.path.exists(model_path):
            print(f"[YOLOe] ⚠️  模型不存在: {model_path}")
            return

        import time as _t; t0 = _t.time()
        print(f"[YOLOe] 加载: {model_path}")
        try:
            from ultralytics import YOLO
            self._model = YOLO(model_path)
            # 读取模型自带的类别表, 检测后过滤到 COCO 80 类
            self._model_names = {}
            if hasattr(self._model, 'names') and self._model.names:
                self._model_names = dict(self._model.names)
                n_classes = len(self._model_names)
                # 统计跟 COCO 80 重叠多少
                coco_hit = sum(1 for nm in self._model_names.values()
                               if _resolve_coco_id(nm) > 0)
                print(f"[YOLOe] 模型类别: {n_classes} 类, "
                      f"其中 {coco_hit} 个可映射到 COCO 80")
            else:
                print(f"[YOLOe] ⚠️  模型无 names 属性, 将用 result.names 获取")
            print(f"[YOLOe] 检测后过滤: 仅保留 COCO 80 类 (非 80 类自动丢弃)")
            elapsed = _t.time() - t0
            print(f"[YOLOe] ✅ 加载完成 ({elapsed:.1f}s)  "
                  f"conf={self.conf}  imgsz={self.imgsz}  "
                  f"max_rois={self.max_rois}")
            self.is_loaded = True
        except Exception as e:
            print(f"[YOLOe] ❌ 加载失败: {e}")

    def has_image(self, img_id) -> bool:
        """检查缓存中是否已检测过该图片。"""
        return str(img_id) in self._cache

    def detect(self, image: Image.Image,
               img_id: Optional[str] = None,
               max_per_cat: int = 2) -> Optional[list]:
        """
        对一张 PIL Image 运行 YOLOe 检测, 返回 ROI 列表.
        格式与 CocoValBoxLoader.get_rois() 完全一致:
          [{'bbox': [x1,y1,x2,y2], 'category_id': int,
            'name': str, 'confidence': float, 'is_core': bool}, ...]
        坐标归一化到 [0,1].
        """
        if not self.is_loaded or self._model is None:
            return None

        # 检查缓存
        iid = str(img_id) if img_id else None
        if iid and iid in self._cache:
            return self._cache[iid]

        try:
            results = self._model.predict(
                image, conf=self.conf, imgsz=self.imgsz,
                verbose=False)
        except Exception as e:
            print(f"[YOLOe] 检测失败: {e}")
            return None

        if not results or len(results) == 0:
            if iid:
                self._cache[iid] = None
            return None

        result = results[0]
        if result.boxes is None or len(result.boxes) == 0:
            if iid:
                self._cache[iid] = None
            return None

        W, H = image.size
        det_boxes = result.boxes
        # 按置信度降序排列
        confs = det_boxes.conf.cpu().numpy()
        sort_idx = np.argsort(-confs)

        raw_list = []
        for idx in sort_idx:
            # xyxy 格式
            xyxy = det_boxes.xyxy[idx].cpu().numpy()
            x1_px, y1_px, x2_px, y2_px = xyxy
            conf_val = float(confs[idx])

            # 获取类别 — 从模型自带 names 表获取, 再映射到 COCO 80
            cls_idx = int(det_boxes.cls[idx].cpu().item())
            # 优先用 result.names (每次推理结果自带), 其次用加载时缓存的
            if hasattr(result, 'names') and result.names:
                det_name = result.names.get(cls_idx, 'unknown')
            elif self._model_names:
                det_name = self._model_names.get(cls_idx, 'unknown')
            else:
                det_name = 'unknown'

            # 解析为 COCO category_id (0=不在 COCO 80 类中)
            coco_id = _resolve_coco_id(det_name)
            # ★ 过滤: 只保留 COCO 80 类, 非 80 类直接跳过
            if coco_id == 0:
                continue
            coco_name = _COCO_ID_TO_NAME.get(coco_id, det_name)

            # 归一化到 [0, 1]
            x1 = max(0.0, x1_px / W)
            y1 = max(0.0, y1_px / H)
            x2 = min(1.0, x2_px / W)
            y2 = min(1.0, y2_px / H)

            # 过滤太小的 box
            if x2 - x1 < 0.01 or y2 - y1 < 0.01:
                continue

            raw_list.append({
                'bbox': [x1, y1, x2, y2],
                'category_id': coco_id,
                'name': coco_name,
                'confidence': conf_val,
                'area': (x2 - x1) * (y2 - y1),
            })

        if not raw_list:
            if iid:
                self._cache[iid] = None
            return None

        # 按面积降序 (跟 CocoValBoxLoader 逻辑一致)
        raw_list.sort(key=lambda a: a['area'], reverse=True)

        # 每类最多 max_per_cat 个, 总共最多 max_rois 个
        boxes, cat_cnt, overflow = [], {}, []
        for det in raw_list:
            cid = det['category_id']
            cnt = cat_cnt.get(cid, 0)
            if cnt < max_per_cat:
                box = {
                    'bbox': det['bbox'],
                    'category_id': cid,
                    'name': det['name'],
                    'confidence': det['confidence'],
                    'is_core': len(boxes) < self.core_count,
                }
                boxes.append(box)
                cat_cnt[cid] = cnt + 1
                if len(boxes) >= self.max_rois:
                    break
            else:
                overflow.append(det)
        for det in overflow:
            if len(boxes) >= self.max_rois:
                break
            box = {
                'bbox': det['bbox'],
                'category_id': det['category_id'],
                'name': det['name'],
                'confidence': det['confidence'],
                'is_core': len(boxes) < self.core_count,
            }
            boxes.append(box)

        result_boxes = boxes if boxes else None

        # 缓存
        if iid:
            self._cache[iid] = result_boxes

        return result_boxes

    def get_rois(self, img_id) -> Optional[list]:
        """兼容接口: 从缓存获取已检测的 ROI (需先调 detect)。"""
        iid = str(img_id)
        return self._cache.get(iid, None)


# ================================================================
# [v6.2.1] Checkpoint 尺寸/结构检测
# ================================================================

def _detect_sizes_from_ckpt(state_dict: dict) -> dict:
    """
    检测 checkpoint 中实际存在的模块尺寸 (v6.2.1 沿用 v6.2 逻辑)。

    v6.2.1 关键 (相对训练管线):
      - 视觉编码器: LAVIS eva_vit_g.pth (BLIP-2 官方)
      - Token 布局: 动态 "32vis + N流" (不再固定 52)
      - INIT-LAVIS: Q-Former cross-attn K/V ← LAVIS eva_vit_g.pth
      - per-query-token 检索 (32q 独立 top-5 合并去重)
      - V6.6.1 model.py + V6.9.3 retriever.py (ROI 受限 attr 检索)
    """
    sizes = {}

    # Q-Former query_tokens 检测 (ckpt 旧名 → model 新属性名映射)
    # [SIGLIP-LEGACY-RENAMED] 旧 ckpt 里 'siglip_qformer.*' 对应新模型 'aux_qformer.*'
    #                         CaptionModel.load_state_dict() 会自动 remap
    _qf_ckpt_to_attr = [
        ('evaclip_qformer',         'evaclip_qformer'),
        ('siglip_qformer',          'aux_qformer'),  # ← 旧 ckpt 兼容
        ('aux_qformer',             'aux_qformer'),  # ← 新 ckpt
        ('final_qformer',           'final_qformer'),
        ('evcap_qformer',           'evcap_qformer'),
        ('evcap_retrieval_qformer', 'evcap_retrieval_qformer'),
        ('fusion_qformer',          'fusion_qformer'),
    ]
    for ckpt_prefix, attr_name in _qf_ckpt_to_attr:
        k = f'{ckpt_prefix}.query_tokens'
        if k in state_dict:
            sizes[f'{attr_name}_nq'] = state_dict[k].shape[1]
            tag = f" [LEGACY→{attr_name}]" if ckpt_prefix != attr_name else ""
            print(f"  [ckpt] {ckpt_prefix}.query_tokens: nq={state_dict[k].shape[1]}{tag}")

    # LightSoftRetriever (v6.2 改为多流 streams)
    has_soft = any('soft_retriever.' in k for k in state_dict)
    print(f"  [ckpt] LightSoftRetriever: {'存在' if has_soft else '不存在'}"
          f"  (per-query-token 检索)")

    # ── CrossViTExchange ─────────────────────────────────────
    has_cross_vit = any('cross_vit_exchange.' in k for k in state_dict)
    if has_cross_vit:
        n_keys = sum(1 for k in state_dict if k.startswith('cross_vit_exchange.'))
        out_v2r_w = state_dict.get('cross_vit_exchange.out_v2r.weight')
        zero_init = (out_v2r_w is not None
                     and out_v2r_w.abs().max().item() < 1e-4)
        print(f"  [ckpt] cross_vit_exchange: {n_keys} keys"
              f"  {'(zero-init, 未训练)' if zero_init else '(已训练)'}")
    else:
        print(f"  [ckpt] cross_vit_exchange: 不存在 (将用 zero-init + BLIP2 q/k/v 初始化)")

    # ── AdaptiveMergeGate ─────────────────────────────────────
    has_merge_gate = any('adaptive_merge_gate.' in k for k in state_dict)
    if has_merge_gate:
        n_keys = sum(1 for k in state_dict if k.startswith('adaptive_merge_gate.'))
        bias_key = 'adaptive_merge_gate.global_bias'
        if bias_key in state_dict:
            bias_val = float(state_dict[bias_key].item())
            gate_init = torch.sigmoid(torch.tensor(bias_val)).item()
            print(f"  [ckpt] adaptive_merge_gate: {n_keys} keys"
                  f"  global_bias={bias_val:.3f} → gate≈{gate_init:.3f}"
                  f" (vis={gate_init:.1%}, roi={1-gate_init:.1%})")
        else:
            print(f"  [ckpt] adaptive_merge_gate: {n_keys} keys (无 global_bias)")
    else:
        print(f"  [ckpt] adaptive_merge_gate: 不存在"
              f"  (将用 gate_mlp←Normal(0,0.01) + global_bias=0.6 初始化)")

    # ── NameToVisualPrompt ────────────────────────────────────
    has_name_vis_prompt = any('name_to_vis_prompt.' in k for k in state_dict)
    if has_name_vis_prompt:
        n_keys = sum(1 for k in state_dict if k.startswith('name_to_vis_prompt.'))
        gate_key = 'name_to_vis_prompt.gate'
        if gate_key in state_dict:
            gate_val = float(state_dict[gate_key].item())
            gate_sigmoid = torch.sigmoid(torch.tensor(gate_val)).item()
            print(f"  [ckpt] name_to_vis_prompt: {n_keys} keys"
                  f"  gate={gate_val:.2f} → sigmoid={gate_sigmoid:.3f}")
        else:
            print(f"  [ckpt] name_to_vis_prompt: {n_keys} keys")
    else:
        print(f"  [ckpt] name_to_vis_prompt: 不存在"
              f"  (将用 gate=−2.0, proj←Normal(0,0.02) 初始化)")

    # ── roi_to_retrieval_proj eye() 检测 ──────────────────────
    has_roi_ret = any('roi_to_retrieval_proj.' in k for k in state_dict)
    if has_roi_ret:
        n_keys = sum(1 for k in state_dict if k.startswith('roi_to_retrieval_proj.'))
        linear_w = state_dict.get('roi_to_retrieval_proj.1.weight')
        if linear_w is not None:
            eye_ref = torch.eye(linear_w.shape[0])
            is_eye = (linear_w.cpu() - eye_ref).abs().mean().item() < 0.05
            print(f"  [ckpt] roi_to_retrieval_proj: {n_keys} keys"
                  f"  {'(仍为 eye() 恒等)' if is_eye else '(已训练偏离 eye())'}")
        else:
            print(f"  [ckpt] roi_to_retrieval_proj: {n_keys} keys")
    else:
        print(f"  [ckpt] roi_to_retrieval_proj: 不存在 (将用 eye() 初始化)")

    # ── TriStreamFusion ───────────────────────────────────────
    has_tri      = any('tri_stream_fusion.' in k for k in state_dict)
    has_tri_attr = any('tri_stream_fusion.q3.' in k for k in state_dict)
    has_tri_name = any('tri_stream_fusion.q2.' in k for k in state_dict)
    print(f"  [ckpt] TriStreamFusion: {'存在' if has_tri else '不存在'}"
          f"{' (+name-stream q2)' if has_tri_name else ''}"
          f"{' (+attr-stream q3)' if has_tri_attr else ''}")

    # [compat] tri_stream_fusion.q1r
    has_tri_q1r = any('tri_stream_fusion.q1r.' in k for k in state_dict)
    if has_tri_q1r:
        n_compat = sum(1 for k in state_dict if any(
            k.startswith(f'tri_stream_fusion.{p}')
            for p in ['q1r.', 'k1r.', 'v1r.', 'out1r.', 'norm1r.']))
        print(f"  [ckpt] [compat] tri_stream_fusion.q1r/k1r/v1r/out1r/norm1r:"
              f" {n_compat} keys (已迁至 cross_vit_exchange, 冻结)")
    else:
        print(f"  [ckpt] [compat] tri_stream_fusion CrossViT(q1r…): 不存在")

    # merge_alpha
    has_merge_alpha = 'tri_stream_fusion.merge_alpha' in state_dict
    if has_merge_alpha:
        val = float(state_dict['tri_stream_fusion.merge_alpha'].item())
        print(f"  [ckpt] tri_stream_fusion.merge_alpha: {val:.4f}"
              f" (sigmoid={torch.sigmoid(torch.tensor(val)).item():.4f}, 已冻结)")
    else:
        print(f"  [ckpt] tri_stream_fusion.merge_alpha: 不存在 (将用默认 0.4)")

    # evcap_qformer.query_in_proj (forward_joint)
    has_forward_joint = 'evcap_qformer.query_in_proj.weight' in state_dict
    print(f"  [ckpt] evcap_qformer.query_in_proj (forward_joint):"
          f" {'存在' if has_forward_joint else '不存在'}")

    # roi_token_aggregator
    has_roi_agg = any('roi_token_aggregator.' in k for k in state_dict)
    if has_roi_agg:
        n_keys = sum(1 for k in state_dict if k.startswith('roi_token_aggregator.'))
        print(f"  [ckpt] roi_token_aggregator: {n_keys} keys")
    else:
        print(f"  [ckpt] roi_token_aggregator: 不存在 (将用 BLIP2 self-attn 初始化)")

    # roi_align_pooler.edge_sharpness
    has_edge_sharp = 'roi_align_pooler.edge_sharpness' in state_dict
    if has_edge_sharp:
        val = float(state_dict['roi_align_pooler.edge_sharpness'].item())
        print(f"  [ckpt] roi_align_pooler.edge_sharpness: {val:.2f}")
    else:
        print(f"  [ckpt] roi_align_pooler.edge_sharpness: 不存在 (将用默认值 3.0)")

    # [compat] fusion_qformer / roi_to_1408_proj
    has_fusion_qf = any('fusion_qformer.' in k for k in state_dict)
    if has_fusion_qf:
        n_keys = sum(1 for k in state_dict if k.startswith('fusion_qformer.'))
        print(f"  [ckpt] fusion_qformer: {n_keys} keys")
    else:
        print(f"  [ckpt] fusion_qformer: 不存在 (将用 BLIP2 初始化)")

    has_roi_1408 = any('roi_to_1408_proj.' in k for k in state_dict)
    if has_roi_1408:
        n_keys = sum(1 for k in state_dict if k.startswith('roi_to_1408_proj.'))
        print(f"  [ckpt] [compat] roi_to_1408_proj: {n_keys} keys")
    else:
        print(f"  [ckpt] [compat] roi_to_1408_proj: 不存在")

    # layer_scale
    has_layer_scale = 'layer_scale' in state_dict
    if has_layer_scale:
        val = float(state_dict['layer_scale'].item())
        print(f"  [ckpt] layer_scale: {val:.4f}")
    else:
        print(f"  [ckpt] layer_scale: 不存在 (将用默认 0.3)")

    # vis_enhancer (新名) / siglip_aligner / evaclip_aligner (旧名)
    # [SIGLIP-LEGACY-RENAMED] 旧 ckpt 'siglip_aligner.*' 对应新模型 'vis_enhancer.*'
    has_vis_enhancer    = any('vis_enhancer.' in k for k in state_dict)
    has_evaclip_aligner = any('evaclip_aligner.' in k for k in state_dict)
    has_siglip_aligner  = any('siglip_aligner.' in k for k in state_dict)
    if has_vis_enhancer:
        print(f"  [ckpt] vis_enhancer: 存在 (新名)")
    elif has_evaclip_aligner:
        print(f"  [ckpt] evaclip_aligner → vis_enhancer: 旧名存在 (将自动映射)")
    elif has_siglip_aligner:
        print(f"  [ckpt] siglip_aligner → vis_enhancer: 旧名存在 (将自动映射)")
    else:
        print(f"  [ckpt] vis_enhancer: 不存在")

    # 汇总标志
    sizes['_has_cross_vit_exchange']  = has_cross_vit
    sizes['_has_adaptive_merge']      = has_merge_gate
    sizes['_has_name_vis_prompt']     = has_name_vis_prompt
    sizes['_has_merge_alpha']         = has_merge_alpha
    sizes['_has_tri_q1r_compat']      = has_tri_q1r
    sizes['_has_forward_joint']       = has_forward_joint
    sizes['_has_roi_agg']             = has_roi_agg
    sizes['_has_fusion_qf']           = has_fusion_qf
    sizes['_has_roi_1408']            = has_roi_1408
    sizes['_has_roi_ret']             = has_roi_ret
    sizes['_has_edge_sharp']          = has_edge_sharp
    sizes['_has_layer_scale']         = has_layer_scale
    sizes['_has_tri_attr']            = has_tri_attr
    sizes['_has_tri_name']            = has_tri_name
    sizes['_has_vis_enhancer']        = has_vis_enhancer
    sizes['_has_evaclip_aligner']     = has_evaclip_aligner
    sizes['_has_siglip_aligner']      = has_siglip_aligner
    sizes['_has_soft_retriever']      = has_soft
    return sizes


def _patch_model_structure(model, sizes: dict):
    """修正 Q-Former 的 query_tokens 数量。"""
    device = next(model.parameters()).device
    D = 768
    # [SIGLIP-LEGACY-RENAMED] 用模型实际属性名 (aux_qformer 替代 siglip_qformer)
    for prefix in ('evaclip_qformer', 'aux_qformer',
                   'final_qformer',
                   'evcap_qformer', 'evcap_retrieval_qformer',
                   'fusion_qformer'):
        key = f'{prefix}_nq'
        if key not in sizes: continue
        target_nq = sizes[key]
        mod = getattr(model, prefix, None)
        if mod is None or not hasattr(mod, 'query_tokens'): continue
        current_nq = mod.query_tokens.shape[1]
        if current_nq != target_nq:
            mod.query_tokens = nn.Parameter(
                torch.zeros(1, target_nq, D, device=device))
            if hasattr(mod, 'num_queries'):
                mod.num_queries = target_nq
            print(f"  ✓ {prefix}.query_tokens: {current_nq} → {target_nq}")
    return model


# ================================================================
# 工具函数
# ================================================================

def _ensure_str(cap) -> str:
    if isinstance(cap, str):   return cap.strip()
    if isinstance(cap, list):
        for item in cap:
            s = _ensure_str(item)
            if s: return s
        return ""
    if isinstance(cap, dict):
        return str(cap.get('caption', cap.get('raw', cap.get('text', '')))).strip()
    return str(cap).strip()


def _extract_image_id_from_filename(fname: str) -> Optional[int]:
    base = os.path.basename(fname)
    m = re.search(r'(\d{12})', base)
    if m: return int(m.group(1))
    m = re.match(r'^(\d+)\.\w+$', base)
    if m: return int(m.group(1))
    return None


def _get_token_layout(model) -> str:
    """[v6.2.1] 动态 Token 布局: 32vis + N流 (从 soft_retriever.streams 读取)"""
    try:
        if (hasattr(model, 'soft_retriever')
                and hasattr(model.soft_retriever, 'streams')):
            streams = model.soft_retriever.streams
            n_streams = len(streams)
            stream_names = '+'.join(streams[:5])
            if n_streams > 5:
                stream_names += f'+...({n_streams}流)'
            return f"32vis + {n_streams}流({stream_names})"
        return "32vis"
    except Exception:
        return "32vis"


# ================================================================
# 权重加载分析
# ================================================================

def _analyze_checkpoint_vs_model(ckpt_state: dict, model_state: dict) -> dict:
    ckpt_keys  = set(ckpt_state.keys())
    model_keys = set(model_state.keys())

    def grp(keys):
        g = defaultdict(list)
        for k in keys: g[k.split('.')[0]].append(k)
        return g

    return {
        'common':            ckpt_keys & model_keys,
        'only_in_ckpt':      ckpt_keys - model_keys,
        'only_in_model':     model_keys - ckpt_keys,
        'ckpt_only_groups':  grp(ckpt_keys  - model_keys),
        'model_only_groups': grp(model_keys - ckpt_keys),
        'common_groups':     grp(ckpt_keys  & model_keys),
    }


def _print_load_report(analysis: dict, sizes: dict, verbose: bool = True):
    """[v6.2.1] 权重加载分析报告。"""
    common        = analysis['common']
    only_in_ckpt  = analysis['only_in_ckpt']
    only_in_model = analysis['only_in_model']

    print(f"\n{'='*70}")
    print(f"权重加载分析报告 (caption_model v6.4 + model v6.6.1 + train v6.2.1)")
    print(f"{'='*70}")
    print(f"  可加载 (共有):     {len(common):5d} keys")
    print(f"  checkpoint 独有:   {len(only_in_ckpt):5d} keys (忽略)")
    print(f"  model 独有:        {len(only_in_model):5d} keys (随机/默认初始化)")

    print(f"\n  v6.2.1 层状态:")
    for tag, key, desc, init in [
        ('v6.2.1-VIS', None,                    '视觉编码器 LAVIS eva_vit_g.pth',              'load_vision_encoders() 加载'),
        ('v6.2.1-QF',  None,                    'Q-Former cross-attn K/V ← LAVIS',              'INIT-LAVIS 自动触发'),
        ('v5.9.12-A',  '_has_edge_sharp',        'roi_align_pooler.edge_sharpness',             '默认值 3.0'),
        ('v5.9.12-C',  '_has_roi_ret',           'roi_to_retrieval_proj',                       'eye() 恒等初始化'),
        ('v5.9.12-D',  '_has_tri_attr',          'tri_stream_fusion (attr stream q3)',          'xavier'),
        ('v5.9.12-G',  '_has_forward_joint',     'evcap_qformer.query_in_proj (joint)',         'xavier'),
        ('v5.9.12-G',  '_has_roi_agg',           'roi_token_aggregator',                        'BLIP2 self-attn'),
        ('v5.9.13-A',  '_has_cross_vit_exchange','cross_vit_exchange',                          'BLIP2 q/k/v + zero-init out'),
        ('v5.9.13-B',  '_has_adaptive_merge',    'adaptive_merge_gate',                         'Normal(0,0.01)+bias=0.6'),
        ('v5.9.16',    '_has_name_vis_prompt',   'name_to_vis_prompt (ViPCap 风格)',            'gate=−2.0'),
        ('v5.9.16',    '_has_layer_scale',       'layer_scale',                                 '默认 0.3'),
        ('v6.2.1',     '_has_soft_retriever',    'soft_retriever (per-query 检索, 多流 bank)',    '多流余弦检索'),
        ('compat',     '_has_tri_q1r_compat',    'tri_stream_fusion.q1r/k1r (遗留)',            '已冻结/xavier'),
        ('compat',     '_has_fusion_qf',         'fusion_qformer',                              'BLIP2 + LAVIS'),
        ('compat',     '_has_roi_1408',          'roi_to_1408_proj',                             'xavier'),
    ]:
        if key is None:
            status = f"⏳ {init}"
        else:
            has = sizes.get(key, False)
            status = f"✅ 从 ckpt 加载" if has else f"⚠️  ckpt 无此层 → {init} 初始化"
        print(f"    [{tag}] {desc:<52s} {status}")

    if verbose:
        all_modules  = (set(analysis['common_groups'])
                        | set(analysis['ckpt_only_groups'])
                        | set(analysis['model_only_groups']))
        skip_modules = {'vision_frontend'}
        print(f"\n  各模块加载情况:")
        print(f"  {'模块名':<35s} {'加载':>6s} {'忽略':>6s} {'缺失':>6s}  状态")
        print(f"  {'-'*70}")
        for mod in sorted(all_modules):
            if mod in skip_modules: continue
            loaded  = len(analysis['common_groups'].get(mod, []))
            ignored = len(analysis['ckpt_only_groups'].get(mod, []))
            missing = len(analysis['model_only_groups'].get(mod, []))
            if mod == 'decoder':
                missing -= sum(
                    1 for k in analysis['model_only_groups'].get(mod, []) if 'llm' in k)
            if loaded == 0 and ignored == 0 and missing == 0: continue
            if   loaded > 0 and ignored == 0 and missing == 0: status = "✅ 完全加载"
            elif loaded > 0 and missing == 0:                  status = "🟡 部分加载"
            elif loaded == 0 and ignored > 0:                  status = "❌ 完全忽略"
            elif loaded == 0 and missing > 0:                  status = "⚠️  全部缺失"
            else:                                               status = "🟡 部分加载"
            print(f"  {mod:<35s} {loaded:>6d} {ignored:>6d} {missing:>6d}  {status}")
    print(f"{'='*70}\n")


def _verify_weights(model):
    """[v6.2.1] 验证关键模块权重。"""
    # [SIGLIP-LEGACY-RENAMED] siglip_aligner / evaclip_aligner → vis_enhancer
    #                          siglip_qformer → aux_qformer
    modules = [
        'evaclip_qformer',
        'aux_qformer',                    # [SIGLIP-LEGACY-RENAMED] was siglip_qformer
        'final_qformer',
        'evcap_qformer',
        'fusion_qformer',
        'vis_enhancer',                   # [SIGLIP-LEGACY-RENAMED] was siglip_aligner / evaclip_aligner
        'aoa_gate',
        'roi_name_injector',
        'roi_token_aggregator',
        'tri_stream_fusion',
        'retrieval_qformer',
        'contrastive_head',
        'soft_retriever',
        'roi_to_1408_proj',
        'roi_to_retrieval_proj',
        'roi_align_pooler',
    ]
    print("[权重验证] 检查关键模块 (v6.2.1-YOLOE)...")
    for name in modules:
        mod = getattr(model, name, None)
        if mod is None:
            print(f"  ⚠️  {name}: 模型中无此属性")
            continue
        params = list(mod.parameters())
        if not params: continue
        if hasattr(mod, 'query_tokens'):
            mean_val = mod.query_tokens.abs().mean().item()
            status = "✓" if mean_val > 1e-6 else "⚠️  接近全零"
            print(f"  {status} {name}.query_tokens  mean={mean_val:.4f}")
        elif hasattr(mod, 'edge_sharpness'):
            val = mod.edge_sharpness.item()
            print(f"  ✓ {name}.edge_sharpness = {val:.2f}")
        else:
            mean_val = sum(p.abs().mean().item() for p in params) / len(params)
            status = "✓" if mean_val > 1e-6 else "⚠️"
            n_param = sum(p.numel() for p in params)
            print(f"  {status} {name}: {n_param/1e6:.2f}M  mean={mean_val:.4f}")

    # ── CrossViTExchange ─────────────────────────────────────
    cve = getattr(model, 'cross_vit_exchange', None)
    if cve is not None:
        n_param = sum(p.numel() for p in cve.parameters())
        try:
            out_v2r_mean = cve.out_v2r.weight.abs().mean().item()
            out_r2v_mean = cve.out_r2v.weight.abs().mean().item()
            zero_status  = ("zero-init 未训练" if out_v2r_mean < 1e-4 else "已训练")
            gate_v_bias  = cve.gate_v[1].bias.item() if hasattr(cve.gate_v[1], 'bias') else 0.0
            gate_r_bias  = cve.gate_r[1].bias.item() if hasattr(cve.gate_r[1], 'bias') else 0.0
            gate_v_val   = torch.sigmoid(torch.tensor(gate_v_bias)).item()
            gate_r_val   = torch.sigmoid(torch.tensor(gate_r_bias)).item()
            print(f"  ✓ cross_vit_exchange: {n_param/1e6:.2f}M  ({zero_status})"
                  f"  [INIT-1: q/k/v←BLIP2]")
            print(f"      out_v2r.weight mean={out_v2r_mean:.6f}  "
                  f"out_r2v.weight mean={out_r2v_mean:.6f}")
            print(f"      gate_v bias={gate_v_bias:.3f}→sigmoid={gate_v_val:.3f}  "
                  f"gate_r bias={gate_r_bias:.3f}→sigmoid={gate_r_val:.3f}")
        except Exception:
            print(f"  ✓ cross_vit_exchange: {n_param/1e6:.2f}M (属性路径与预期不同)")
    else:
        print(f"  ⚠️  cross_vit_exchange: 模型中无此属性")

    # ── AdaptiveMergeGate ────────────────────────────────────
    amg = getattr(model, 'adaptive_merge_gate', None)
    if amg is not None:
        n_param = sum(p.numel() for p in amg.parameters())
        try:
            bias_val   = amg.global_bias.item()
            gate_init  = torch.sigmoid(torch.tensor(bias_val)).item()
            gate_last_mean = amg.gate_mlp[3].weight.abs().mean().item()
            zero_status    = ("gate_mlp 未训练" if gate_last_mean < 1e-5 else "已训练")
            print(f"  ✓ adaptive_merge_gate: {n_param/1e6:.2f}M  ({zero_status})"
                  f"  [INIT-2: gate_mlp←Normal(0,0.01)]")
            print(f"      global_bias={bias_val:.3f} → gate≈{gate_init:.3f}"
                  f" (初始 vis={gate_init:.1%}, roi={1-gate_init:.1%})")
            print(f"      gate_mlp[-1].weight mean={gate_last_mean:.6f}")
        except Exception:
            print(f"  ✓ adaptive_merge_gate: {n_param/1e6:.2f}M (属性路径与预期不同)")
    else:
        print(f"  ⚠️  adaptive_merge_gate: 模型中无此属性")

    # ── NameToVisualPrompt ────────────────────────────────────
    nvp = getattr(model, 'name_to_vis_prompt', None)
    if nvp is not None:
        n_param = sum(p.numel() for p in nvp.parameters())
        try:
            gate_val = nvp.gate.item()
            gate_sigmoid = torch.sigmoid(torch.tensor(gate_val)).item()
            print(f"  ✓ name_to_vis_prompt: {n_param/1e6:.2f}M"
                  f"  [INIT-3: gate={gate_val:.2f}→sigmoid={gate_sigmoid:.3f}]")
        except Exception:
            print(f"  ✓ name_to_vis_prompt: {n_param/1e6:.2f}M")
    else:
        print(f"  ⚠️  name_to_vis_prompt: 模型中无此属性")

    # ── roi_to_retrieval_proj eye() ──────────────────────────
    rrp = getattr(model, 'roi_to_retrieval_proj', None)
    if rrp is not None:
        try:
            w = rrp[1].weight
            eye_ref = torch.eye(w.shape[0], device=w.device)
            deviation = (w - eye_ref).abs().mean().item()
            is_eye = deviation < 0.05
            print(f"  ✓ roi_to_retrieval_proj  [INIT-4: eye()]"
                  f"  deviation={deviation:.6f}"
                  f"  {'(仍为恒等)' if is_eye else '(已训练偏离)'}")
        except Exception:
            print(f"  ✓ roi_to_retrieval_proj: 存在")

    # ── layer_scale ──────────────────────────────────────────
    if hasattr(model, 'layer_scale'):
        val = model.layer_scale.item()
        print(f"  ✓ layer_scale: {val:.4f}")

    # ── tri_stream_fusion.merge_alpha ────────────────────────
    tri = getattr(model, 'tri_stream_fusion', None)
    if tri is not None and hasattr(tri, 'merge_alpha'):
        raw   = float(tri.merge_alpha.item())
        alpha = torch.sigmoid(torch.tensor(raw)).item()
        grad  = tri.merge_alpha.requires_grad
        print(f"  ✓ tri_stream_fusion.merge_alpha: raw={raw:.4f} → α={alpha:.4f}"
              f" (vis={alpha:.1%}, roi={1-alpha:.1%})"
              f"  {'可训练' if grad else '已冻结 ✓'}")

    # ── LightSoftRetriever memory bank (v6.2.1: 多流) ──────────
    sr = getattr(model, 'soft_retriever', None)
    if sr is not None:
        bank = getattr(sr, '_bank', None)
        streams_attr = getattr(sr, 'streams', None)
        if bank:
            streams    = list(bank.keys())
            sizes_dict = {s: bank[s]['keys'].shape[0] for s in streams}
            print(f"  ✓ soft_retriever.memory_bank: 流={streams}  大小={sizes_dict}")
        elif streams_attr:
            print(f"  ⚠️  soft_retriever.streams={streams_attr}  但 _bank 未设置")
        else:
            print(f"  ⚠️  soft_retriever: memory_bank 未设置")

    # ── Prompt stage 确认 ────────────────────────────────────
    ps   = getattr(model.decoder, '_prompt_stage', 'unknown')
    tmpl = getattr(model.decoder, '_INSTRUCT_TEMPLATES', {}).get(ps, '?')
    print(f"  ✓ decoder prompt stage: {ps}")
    print(f"  ✓ prompt 模板: \"{tmpl.strip() if isinstance(tmpl, str) else tmpl}\"")

    # ── v6.2.1-YOLOE 特性总结 ─────────────────────────────────
    print(f"  ✓ 投票阈值: VOTE_THRESH={_VOTE_THRESH}")
    print(f"  ✓ EVAL gen: max_new={_EVAL_GEN_KWARGS['max_new_tokens']} "
          f"min={_EVAL_GEN_KWARGS['min_new_tokens']} "
          f"beams={_EVAL_GEN_KWARGS['num_beams']} "
          f"rep={_EVAL_GEN_KWARGS['repetition_penalty']} "
          f"no_rep_ngram={_EVAL_GEN_KWARGS['no_repeat_ngram_size']}")
    print(f"  ✓ Token 布局: {_get_token_layout(model)} (动态, 32vis + 多流)")
    print(f"  ✓ [v6.2.1-VIS] 视觉: LAVIS eva_vit_g.pth (与 ext_memory_lvis.pkl 同源)")
    print(f"  ✓ [INIT-LAVIS] cross-attn K/V ← LAVIS eva_vit_g.pth"
          f" (load_vision_encoders 后触发)")
    print(f"  ✓ [PER-Q] per-query-token 检索 (32 query 独立 top-5 合并去重)")
    print(f"  ✓ [YOLOE] ROI 来源: YOLOe > grid (无 GT 泄露, free setup)")
    print("  验证完成\n")


# ================================================================
# 投票诊断格式化
# ================================================================

def _format_vote_topk(vote_data, max_show: int = 10) -> str:
    if vote_data is None:
        return "[无数据]"

    # 旧版双通道 dict (向后兼容)
    if isinstance(vote_data, dict):
        large_items = vote_data.get('large', [])
        small_items = vote_data.get('small', [])
        large_str = ", ".join(f"{n}:{c}" for n, c in large_items[:max_show]) or "—"
        small_str = ", ".join(f"{n}:{c}" for n, c in small_items[:max(max_show//2, 4)]) or "—"
        return f"主[{large_str}] | 小[{small_str}]"

    if not vote_data:
        return "[无投票]"

    items = []
    for entry in vote_data[:max_show]:
        if isinstance(entry, (list, tuple)) and len(entry) == 2:
            name, val = entry
            if isinstance(val, float):
                items.append(f"{name}({val:.2f})")
            else:
                items.append(f"{name}:{val}")
        elif isinstance(entry, str):
            items.append(entry)
    return f"[{', '.join(items)}]" if items else "[无]"


def _format_vote_denoised(denoised_list, thresh: int = _VOTE_THRESH,
                          max_show: int = 8) -> str:
    if not denoised_list:
        return f"[无] (全部被去噪过滤)"
    valid = [n for n in denoised_list if n and n.strip()][:max_show]
    if not valid:
        return "[无有效名称]"
    return f"[{', '.join(valid)}]  ({len(valid)}个)"


# ================================================================
# Karpathy Test Split 加载
# ================================================================

def load_karpathy_test(json_path: str) -> Tuple[List[str], Dict, Dict]:
    print(f"[Karpathy] 加载: {json_path}")
    with open(json_path, encoding='utf-8') as f:
        raw = json.load(f)

    captions:   Dict[str, List[str]] = {}
    id_to_file: Dict[str, str]       = {}

    if isinstance(raw, list):
        items = raw
    elif isinstance(raw, dict):
        items = raw.get('images', raw.get('annotations',
                list(raw.values()) if raw else []))
    else:
        return [], {}, {}

    has_split = items and isinstance(items[0], dict) and 'split' in items[0]

    for item in items:
        if not isinstance(item, dict): continue
        if has_split and item.get('split', '') not in ('test', 'val'): continue

        img_id = item.get('cocoid') or item.get('image_id') or item.get('id')
        fname  = (item.get('file_name') or item.get('filename')
                  or item.get('image', '') or item.get('file_path', ''))

        if img_id is None and fname:
            img_id = _extract_image_id_from_filename(fname)
        if img_id is None: continue

        iid = str(int(img_id))
        id_to_file[iid] = (os.path.basename(fname) if fname
                           else f"COCO_val2014_{int(img_id):012d}.jpg")

        raw_caps = []
        for s in item.get('sentences', []):
            raw_caps.append(
                s.get('raw', s.get('caption', s.get('text', '')))
                if isinstance(s, dict) else s)
        if not raw_caps:
            cap_field = item.get('caption', item.get('captions', []))
            raw_caps = ([cap_field] if isinstance(cap_field, str)
                        else (cap_field if isinstance(cap_field, list) else []))

        flat = [_ensure_str(c) for c in raw_caps if _ensure_str(c)]
        if flat:
            captions[iid] = captions.get(iid, []) + flat

    img_ids = sorted(captions.keys(), key=lambda x: int(x))
    print(f"[Karpathy] ✅ {len(img_ids)} 张图, "
          f"{sum(len(v) for v in captions.values())} 条 caption")
    return img_ids, captions, id_to_file


# ================================================================
# 推理引擎 (v6.2.1-YOLOE)
# ================================================================

class InferenceEngine:
    def __init__(self, config, checkpoint_path: str,
                 device: str = "cuda",
                 yoloe_path: Optional[str] = None,
                 yoloe_conf: float = 0.25,
                 yoloe_imgsz: int = 640,
                 verbose: bool = True,
                 show_retrieval_diag: bool = True):
        self.config              = config
        self.device              = torch.device(device)
        self.verbose             = verbose
        self.show_retrieval_diag = show_retrieval_diag

        # ── tokenizer ──────────────────────────────────────────
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(config.paths.llm_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = 'right'

        # ── [YOLOE-ROI] YOLOe 检测器 ──────────────────────────
        max_rois   = getattr(config.training, 'max_rois', 11)
        core_count = getattr(config.training, 'core_rois', 4)
        self.yoloe_loader: Optional[YOLOeBoxLoader] = None
        if yoloe_path:
            self.yoloe_loader = YOLOeBoxLoader(
                yoloe_path, max_rois=max_rois, core_count=core_count,
                conf=yoloe_conf, imgsz=yoloe_imgsz)
            if not self.yoloe_loader.is_loaded:
                self.yoloe_loader = None

        # ── checkpoint 预读 ────────────────────────────────────
        print(f"\n[Infer] 预读 checkpoint: {checkpoint_path}")
        ckpt_raw   = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        state_dict = ckpt_raw.get('model', ckpt_raw)
        print(f"  stage={ckpt_raw.get('stage', 2)}, "
              f"epoch={ckpt_raw.get('epoch', 0)}, "
              f"global_step={ckpt_raw.get('global_step', 'N/A')}, "
              f"keys={len(state_dict)}")

        sizes = _detect_sizes_from_ckpt(state_dict)

        # ── 构建模型 ───────────────────────────────────────────
        print(f"\n[Infer] 构建 CaptionModel (caption_model v6.4 + v6.6.1 model)...")
        self.model = CaptionModel(config)
        self.model.set_tokenizer(self.tokenizer)

        # Q-Former query_tokens 尺寸修正
        self.model = _patch_model_structure(self.model, sizes)

        # ── 预加载 BLIP2 权重 ─────────────────────────────────
        blip2_path = getattr(config.paths, 'blip2_qformer_path', None)
        if blip2_path and os.path.exists(blip2_path):
            print(f"\n[Infer] 预加载 BLIP2 权重: {blip2_path}")
            self.model.load_blip2_pretrained(blip2_path, freeze=True)
            print(f"  ✅ BLIP2 结构已初始化 (INIT-1/5/6)")
            ret_qf_loaded = getattr(
                getattr(self.model, 'retrieval_qformer', None), '_loaded', None)
            if ret_qf_loaded is not None:
                print(f"  RetrievalQFormer._loaded={ret_qf_loaded}")
        else:
            print(f"\n[Infer] ⚠️  未找到 blip2_qformer_path ({blip2_path})，"
                  f"RetrievalQFormer 将使用 fallback 路径")

        # ── 分析加载情况 ───────────────────────────────────────
        model_state = self.model.state_dict()
        analysis = _analyze_checkpoint_vs_model(state_dict, model_state)
        _print_load_report(analysis, sizes, verbose=verbose)

        # LoRA 支持
        if any('lora_' in k for k in state_dict):
            if hasattr(self.model, 'enable_lora'):
                self.model.enable_lora()
                print("  [LoRA] 已启用")
            else:
                print("  [LoRA] ckpt 含 lora_* 权重, 但模型无 enable_lora()")

        # ── 加载训练 checkpoint ────────────────────────────────
        # [SIGLIP-LEGACY-RENAMED] CaptionModel.load_state_dict() 内部会自动
        #                          remap 旧名 (siglip_aligner.* → vis_enhancer.*,
        #                          siglip_qformer.* → aux_qformer.*) 后再加载
        missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
        real_missing    = [k for k in missing    if 'decoder.llm' not in k]
        real_unexpected = [k for k in unexpected if not k.startswith('vision_frontend.')]

        print(f"\n[加载结果]")
        print(f"  成功加载:   {len(state_dict) - len(unexpected)} keys")
        print(f"  missing:    {len(missing)} "
              f"(LLM主体={len(missing)-len(real_missing)}, 其他={len(real_missing)})")
        print(f"  unexpected: {len(unexpected)} "
              f"(vision_frontend={len(unexpected)-len(real_unexpected)}, "
              f"其他={len(real_unexpected)})")

        if real_missing and verbose:
            print(f"\n  非LLM缺失权重 ({len(real_missing)}):")
            for k in sorted(real_missing)[:25]:
                tag = ""
                if 'cross_vit_exchange' in k:
                    tag = " [INIT-1, CrossViT q/k/v←BLIP2 + zero-init out]"
                elif 'adaptive_merge_gate' in k:
                    tag = " [INIT-2, gate_mlp←Normal(0,0.01) + bias=0.6]"
                elif 'name_to_vis_prompt' in k:
                    tag = " [INIT-3, gate=−2.0, proj←Normal(0,0.02)]"
                elif 'roi_to_retrieval_proj' in k:
                    tag = " [INIT-4, eye() 恒等初始化]"
                elif 'fusion_qformer' in k:
                    tag = " [BLIP2 + LAVIS 初始化]"
                elif 'roi_to_1408_proj' in k:
                    tag = " [xavier 初始化]"
                elif 'edge_sharpness' in k:
                    tag = " [默认 3.0]"
                elif 'roi_token_aggregator' in k:
                    tag = " [INIT-6, BLIP2 self-attn 初始化]"
                elif any(s in k for s in ['tri_stream_fusion.q1r', 'tri_stream_fusion.k1r',
                                           'tri_stream_fusion.v1r', 'tri_stream_fusion.out1r',
                                           'tri_stream_fusion.norm1r']):
                    tag = " [compat, 已迁至 cross_vit_exchange]"
                elif 'merge_alpha' in k:
                    tag = " [tri_stream_fusion, 默认 0.4, 已冻结]"
                elif any(s in k for s in ['tri_stream_fusion.q3', 'tri_stream_fusion.k3',
                                           'tri_stream_fusion.v3', 'tri_stream_fusion.norm_attr']):
                    tag = " [attr stream, xavier]"
                elif 'layer_scale' in k:
                    tag = " [默认 0.3]"
                elif 'soft_retriever' in k:
                    tag = " [多流 bank, per-query 检索]"
                print(f"    - {k}{tag}")
            if len(real_missing) > 25:
                print(f"    ... 还有 {len(real_missing)-25} 个")

        self.model.to(self.device)
        self.model.eval()

        # ── [v6.2.1] 视觉编码器 (LAVIS) + INIT-LAVIS ──────────
        print(f"\n[Infer] 加载视觉编码器 (LAVIS eva_vit_g.pth)...")
        print(f"  [v6.2.1-VIS] 与 ext_memory_lvis.pkl / attr_memory.pkl 构建路径一致")
        print(f"  [INIT-LAVIS] load_vision_encoders() 将自动触发:")
        print(f"    Q-Former cross-attn K/V ← LAVIS eva_vit_g.pth 最后层")
        self.model.load_vision_encoders()

        # prompt stage
        if hasattr(self.model, 'decoder') and \
                hasattr(self.model.decoder, 'set_prompt_stage'):
            self.model.decoder.set_prompt_stage("stage2_late")
            ps   = self.model.decoder._prompt_stage
            tmpl = self.model.decoder._INSTRUCT_TEMPLATES.get(ps, '?')
            print(f"  decoder prompt stage → {ps}")
            print(f"  prompt 模板: \"{tmpl.strip() if isinstance(tmpl, str) else tmpl}\"")

        _verify_weights(self.model)

        # ── 状态汇总 ───────────────────────────────────────────
        yoloe_status = (
            f"✅ YOLOe 已加载 (conf={yoloe_conf}, imgsz={yoloe_imgsz})"
            if self.yoloe_loader else "❌ 未加载 (将用 grid fallback)")

        pkl_retriever = getattr(self.model, '_pkl_retriever', None)
        pkl_status = "❌ 未加载"
        if pkl_retriever is not None and getattr(pkl_retriever, '_loaded', False):
            n_names = len(pkl_retriever._name2idx) if hasattr(pkl_retriever, '_name2idx') else '?'
            n_attrs = (pkl_retriever.attr_keys.shape[0]
                       if pkl_retriever.attr_keys is not None else 0)
            pkl_status = f"✅ 名称库 {n_names} 类 | 属性库 {n_attrs} 条"

        sr   = getattr(self.model, 'soft_retriever', None)
        bank = getattr(sr, '_bank', None) if sr else None
        if bank:
            soft_status = f"✅ memory_bank 流={list(bank.keys())}"
        elif sr is not None and hasattr(sr, 'streams'):
            soft_status = f"⚠️  streams={sr.streams}, bank 未构建"
        else:
            soft_status = "⚠️  soft_retriever 未初始化"

        ret_qf     = getattr(self.model, 'retrieval_qformer', None)
        ret_status = ("✅ 标准 BertModel 已加载"
                      if (ret_qf is not None and getattr(ret_qf, '_loaded', False))
                      else "⚠️  未加载 (fallback 路径)")

        # CrossViT 状态
        cve = getattr(self.model, 'cross_vit_exchange', None)
        cve_str = "N/A"
        if cve is not None:
            try:
                gv_bias = cve.gate_v[1].bias.item() if hasattr(cve.gate_v[1], 'bias') else 0.0
                is_trained = cve.is_trained() if hasattr(cve, 'is_trained') else False
                cve_str = (f"gate_v_bias={gv_bias:.3f}"
                           f"  {'已训练' if is_trained else '未训练(zero-init)'}")
            except Exception:
                cve_str = "存在"

        # AdaptiveMergeGate 状态
        amg = getattr(self.model, 'adaptive_merge_gate', None)
        amg_str = "N/A"
        if amg is not None:
            try:
                bias_val  = amg.global_bias.item()
                gate_init = torch.sigmoid(torch.tensor(bias_val)).item()
                amg_str   = (f"global_bias={bias_val:.3f}→gate≈{gate_init:.3f}"
                             f" (vis={gate_init:.1%}, roi={1-gate_init:.1%})")
            except Exception:
                amg_str = "存在"

        # NameToVisualPrompt 状态
        nvp = getattr(self.model, 'name_to_vis_prompt', None)
        nvp_str = "N/A"
        if nvp is not None:
            try:
                gate_val = nvp.gate.item()
                nvp_str = (f"gate={gate_val:.2f}"
                           f"→sigmoid={torch.sigmoid(torch.tensor(gate_val)).item():.3f}")
            except Exception:
                nvp_str = "存在"

        # merge_alpha 状态
        alpha_str = "N/A"
        tri = getattr(self.model, 'tri_stream_fusion', None)
        if tri is not None and hasattr(tri, 'merge_alpha'):
            raw   = float(tri.merge_alpha.item())
            alpha = torch.sigmoid(torch.tensor(raw)).item()
            alpha_str = f"α={alpha:.4f} (冻结, vis={alpha:.1%}, roi={1-alpha:.1%})"

        # Token 布局 (动态)
        token_layout = _get_token_layout(self.model)

        print(f"\n[Infer] ✅ v6.2.1-YOLOE 初始化完成")
        print(f"  caption_model: v6.4 包结构 | model.py: v6.6.1 | train.py: v6.2.1")
        print(f"  架构: LAVIS 视觉对齐 + per-query-token 检索 + V6.6 ROI 受限 attr + 仅 NTP loss")
        print(f"  [v6.2.1-VIS] 视觉: LAVIS eva_vit_g.pth (与 ext_memory_lvis.pkl 同源)")
        print(f"  [INIT-LAVIS] Q-Former cross-attn K/V ← LAVIS eva_vit_g.pth 最后层")
        print(f"  [PER-Q]      per-query-token 检索 (32q 独立 top-5 合并去重)")
        print(f"  [INIT-1]  CrossViTExchange:    {cve_str}")
        print(f"  [INIT-2]  AdaptiveMergeGate:   {amg_str}")
        print(f"  [INIT-3]  NameToVisualPrompt:  {nvp_str}")
        print(f"  [INIT-4]  roi_to_retrieval_proj ← eye()")
        print(f"  tri_stream_fusion.merge_alpha: {alpha_str}")
        print(f"  Token 布局: {token_layout} (动态, 从 soft_retriever.streams 读取)")
        print(f"  ROI pipeline:  bbox→ROIAlign→per-ROI QFormer→ROITokenAggregator")
        print(f"                 →CrossViTExchange→AdaptiveMergeGate→NameToVisualPrompt")
        print(f"                 →TriStream→joint Q-Former→vis_proj→LLM")
        print(f"  YOLOe ROI:     {yoloe_status}")
        print(f"  ROI 来源优先级: YOLOe > grid (无 GT 泄露, free setup)")
        print(f"  PKL 检索器:    {pkl_status}")
        print(f"  SoftRetriever: {soft_status}")
        print(f"  RetrievalQF:   {ret_status}")
        print(f"  EVAL gen:      beams={_EVAL_GEN_KWARGS['num_beams']} "
              f"max_new={_EVAL_GEN_KWARGS['max_new_tokens']} "
              f"min={_EVAL_GEN_KWARGS['min_new_tokens']} "
              f"rep={_EVAL_GEN_KWARGS['repetition_penalty']} "
              f"no_rep_ngram={_EVAL_GEN_KWARGS['no_repeat_ngram_size']}")

    def _restore_cuda(self):
        if self.device.type == 'cuda':
            torch.cuda.set_device(self.device.index or 0)

    def _get_lvis_boxes(self, image: Image.Image,
                        img_id: Optional[str] = None) -> Optional[list]:
        """ROI 来源: YOLOe > None(grid fallback) — 无 GT"""
        if self.yoloe_loader and self.yoloe_loader.is_loaded:
            return self.yoloe_loader.detect(image, img_id=img_id)
        return None

    def _make_generate_kwargs(self, batch: dict) -> dict:
        """
        构建 model.generate() 的 keyword 参数 (v6.2.1).

        [SIGLIP-LEGACY-RENAMED] kwarg 名全部改成 eva_*:
          siglip_patch              → eva_patches
          siglip_mid_patch          → eva_mid_patches
          siglip_rois_for_retrieval → eva_global_per_roi
          siglip_patch_projected    → eva_global_per_roi_alt
          clip_patches              → 已合并入 eva_patches (移除单独 kwarg)
        """
        return dict(
            abs_pos=batch.get('absolute_pos'),
            rel_pos=batch.get('relative_pos'),
            attr_feats=batch.get('attribute_features'),
            tag_sequences=batch.get('tag_sequences'),
            is_core=batch.get('is_core'),
            importance_scores=batch.get('importance_scores'),
            roi_mask=batch.get('roi_mask'),
            category_ids=batch.get('category_ids'),
            eva_patches=batch.get('eva_patches'),
            eva_mid_patches=batch.get('eva_mid_patches'),
            tag_confidence=batch.get('tag_confidence'),
            eva_global_per_roi=batch.get('eva_global_per_roi'),
            eva_global_per_roi_alt=batch.get('eva_global_per_roi_alt'),
            boxes_batch=batch.get('boxes_batch'),
            use_qformer=True,
            use_spatial_anchor=True,
        )

    @torch.no_grad()
    def generate_single(self,
                        image: Image.Image,
                        img_id: Optional[str] = None) -> str:
        lvis_boxes = self._get_lvis_boxes(image, img_id)
        feats = self.model.extract_online([image], lvis_boxes=[lvis_boxes])
        self._restore_cuda()
        batch = {k: v.to(self.device) if torch.is_tensor(v) else v
                 for k, v in feats.items()}
        # [SIGLIP-LEGACY-RENAMED] 位置参数: clip_global/siglip_global/siglip_rois
        #                          → eva_global / eva_global_768 / eva_rois
        with _safe_autocast():
            pred_ids = self.model.generate(
                batch['eva_global'],
                batch['eva_global_768'],
                batch['eva_rois'],
                **self._make_generate_kwargs(batch),
                **_EVAL_GEN_KWARGS,
            )
        return self.model.decode_and_clean(pred_ids[0], self.tokenizer)

    @torch.no_grad()
    def generate_single_with_detail(
            self,
            image: Image.Image,
            img_id: Optional[str] = None,
    ) -> Tuple[str, str, str, str, str, str, str]:
        """
        返回 (caption, retrieval_str, vote_topk_str, vote_denoised_str,
               implicit_str, roi_source, per_query_diag_str)
        roi_source ∈ {'yoloe', 'grid'}
        """
        # ── ROI 来源判断 (YOLOe / grid) ─────────────────────────
        roi_source = 'grid'
        lvis_boxes = None
        if self.yoloe_loader and self.yoloe_loader.is_loaded:
            lvis_boxes = self.yoloe_loader.detect(image, img_id=img_id)
            if lvis_boxes:
                roi_source = 'yoloe'

        feats = self.model.extract_online([image], lvis_boxes=[lvis_boxes])
        self._restore_cuda()
        batch = {k: v.to(self.device) if torch.is_tensor(v) else v
                 for k, v in feats.items()}

        # ── retrieval_vecs (per-query-token, v6.2.1) ─────────────
        # [SIGLIP-LEGACY-RENAMED] siglip_patch → eva_patches,
        #                          siglip_mid_patch → eva_mid_patches
        retrieval_vecs = None
        eva_patches = batch.get('eva_patches')
        if eva_patches is not None:
            try:
                mid_patches = batch.get('eva_mid_patches')
                if (mid_patches is not None
                        and hasattr(self.model, '_fuse_ms_patch')):
                    try:
                        fused = self.model._fuse_ms_patch(eva_patches, mid_patches)
                    except Exception:
                        fused = eva_patches
                else:
                    fused = eva_patches
                _, retrieval_vecs = self.model.retrieval_qformer(fused.float())
            except Exception:
                retrieval_vecs = None

        # ── PKL 检索名称+属性 ──────────────────────────────────
        # [SIGLIP-LEGACY-RENAMED] _collect_retrieval_names_and_attrs() 签名:
        #   siglip_g → eva_global_768, siglip_rois → eva_rois,
        #   siglip_rois_for_retrieval → eva_global_per_roi,
        #   siglip_patch_projected → eva_global_per_roi_alt
        retrieval_str = ""
        try:
            names_batch, attrs_batch = self.model._collect_retrieval_names_and_attrs(
                1,  # B
                category_ids=batch.get('category_ids'),
                tag_sequences=batch.get('tag_sequences'),
                is_core=batch.get('is_core'),
                eva_global_768=batch['eva_global_768'],
                eva_rois=batch['eva_rois'],
                tag_confidence=batch.get('tag_confidence'),
                eva_global_per_roi=batch.get('eva_global_per_roi'),
                eva_global_per_roi_alt=batch.get('eva_global_per_roi_alt'),
                retrieval_vecs=retrieval_vecs,
            )
            if names_batch and names_batch[0]:
                parts = []
                for name in names_batch[0]:
                    attrs = attrs_batch[0].get(name, []) if attrs_batch else []
                    if attrs and isinstance(attrs[0], dict):
                        attr_vals = [a['value'] if isinstance(a, dict) else str(a)
                                     for a in attrs[:2]]
                    else:
                        attr_vals = [str(a) for a in attrs[:2] if a]
                    parts.append(f"{name}({','.join(attr_vals)})" if attr_vals else name)
                retrieval_str = ", ".join(parts)
        except Exception as e:
            retrieval_str = f"[retrieval error: {e}]"

        # ── 生成 caption ────────────────────────────────────────
        # [SIGLIP-LEGACY-RENAMED] 位置参数已重命名
        with _safe_autocast():
            pred_ids = self.model.generate(
                batch['eva_global'],
                batch['eva_global_768'],
                batch['eva_rois'],
                **self._make_generate_kwargs(batch),
                **_EVAL_GEN_KWARGS,
            )

        # ── 投票诊断 ────────────────────────────────────────────
        vote_topk_str = ""
        try:
            vt = getattr(self.model, '_last_vote_topk', None)
            if vt and len(vt) > 0 and vt[0] is not None:
                vote_topk_str = _format_vote_topk(vt[0], max_show=10)
        except Exception:
            pass

        vote_denoised_str = ""
        try:
            vd = getattr(self.model, '_last_vote_denoised', None)
            if vd and len(vd) > 0 and vd[0] is not None:
                vote_denoised_str = _format_vote_denoised(
                    vd[0], thresh=_VOTE_THRESH, max_show=8)
        except Exception:
            pass

        # ── implicit names (evcap name_tokens) ─────────────────
        implicit_str = ""
        try:
            impl = getattr(self.model, '_last_implicit_names', None)
            if impl and len(impl) > 0 and impl[0]:
                valid = [n for n in impl[0][:8] if n]
                implicit_str = ", ".join(valid)
        except Exception:
            pass

        # ── [v6.2.1] per-query-token 检索诊断 ──────────────────
        per_query_diag_str = ""
        if (self.show_retrieval_diag
                and hasattr(self.model, 'format_evcap_retrieval_diag')):
            try:
                diag = self.model.format_evcap_retrieval_diag(b=0, show_queries=8)
                if diag:
                    per_query_diag_str = diag
            except Exception:
                pass

        caption = self.model.decode_and_clean(pred_ids[0], self.tokenizer)
        return (caption, retrieval_str, vote_topk_str, vote_denoised_str,
                implicit_str, roi_source, per_query_diag_str)


# ================================================================
# COCO 指标计算
# ================================================================

def compute_coco_metrics(predictions: Dict[int, str],
                         captions: Dict[str, List[str]]) -> Dict[str, float]:
    try:
        from pycocotools.coco import COCO
        from pycocoevalcap.eval import COCOEvalCap
    except ImportError:
        print("⚠️ 需安装: pip install pycocotools pycocoevalcap")
        return {}

    gt_json = {'images': [], 'annotations': [],
               'type': 'captions', 'info': {}, 'licenses': []}
    ann_id = 1
    for img_id_int in predictions:
        iid_str = str(img_id_int)
        if iid_str not in captions: continue
        gt_json['images'].append({'id': img_id_int})
        for cap in captions[iid_str]:
            cap_str = _ensure_str(cap)
            if cap_str:
                gt_json['annotations'].append(
                    {'id': ann_id, 'image_id': img_id_int, 'caption': cap_str})
                ann_id += 1

    gt_img_ids = {ann['image_id'] for ann in gt_json['annotations']}
    gt_json['images'] = [img for img in gt_json['images'] if img['id'] in gt_img_ids]
    if not gt_json['annotations']:
        return {}

    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(gt_json, f); gt_file = f.name
    pred_list = [{'image_id': k, 'caption': _ensure_str(v)}
                 for k, v in predictions.items() if k in gt_img_ids]
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(pred_list, f); pred_file = f.name

    coco_gt   = COCO(gt_file)
    coco_pred = coco_gt.loadRes(pred_file)
    ev = COCOEvalCap(coco_gt, coco_pred)
    ev.params['image_id'] = coco_pred.getImgIds()
    ev.evaluate()
    os.unlink(gt_file); os.unlink(pred_file)
    return dict(ev.eval)


# ================================================================
# 详细打印 (v6.2.1 — 含 per-query 检索块)
# ================================================================

def _print_step_detail(
    step: int,
    iid: str,
    fname: str,
    caption: str,
    gt_caps: List[str],
    retrieval_str: str,
    vote_topk_str: str,
    vote_denoised_str: str,
    implicit_str: str,
    roi_source: str,
    elapsed_ms: float,
    per_query_diag_str: str = "",
    vote_thresh: int = _VOTE_THRESH,
):
    roi_icons = {'yoloe': '🔍 YOLOe-ROI', 'grid': '⬜ grid-ROI'}
    roi_tag   = roi_icons.get(roi_source, roi_source)
    border    = "─" * 72

    lines = [
        f"\n{border}",
        f"[Step {step:5d}]  {fname}  (id={iid})  {roi_tag}  [{elapsed_ms:.0f}ms]",
        border,
    ]

    if gt_caps:
        lines.append(f"  GT ({min(len(gt_caps), 5)}/{len(gt_caps)} 条):")
        for gi, gt in enumerate(gt_caps[:5], 1):
            lines.append(f"    GT-{gi}: {_ensure_str(gt)}")
    else:
        lines.append("  GT: (无)")

    lines.append(f"  PKL检索:            {retrieval_str[:120] if retrieval_str else '(无)'}")
    lines.append(f"  检索 Step1:         {vote_topk_str[:120] if vote_topk_str else '(无)'}")
    lines.append(f"  检索 Step2 去噪后:  "
                 f"{vote_denoised_str[:120] if vote_denoised_str else '(无/全过滤)'}")
    if per_query_diag_str:
        lines.append(f"  [PER-Q] per-query-token 检索:")
        for ln in per_query_diag_str.splitlines():
            lines.append(f"    {ln}")
    lines.append(f"  name_tokens [evcap]:{implicit_str[:120] if implicit_str else '(无)'}")
    lines.append(f"  预测: {caption if caption else '[空/乱码已过滤]'}")
    lines.append(border)

    tqdm.write("\n".join(lines))
    return "\n".join(lines)


# ================================================================
# eval 模式
# ================================================================

def run_eval(
    engine: InferenceEngine,
    image_dir: str,
    karpathy_json: str,
    output_file: str = 'eval_results.json',
    max_images: Optional[int] = None,
    print_every: int = 100,
    show_inject: bool = True,
):
    img_ids, captions, id_to_file = load_karpathy_test(karpathy_json)
    if not img_ids:
        print("❌ 没有加载到任何测试图片"); return {}

    if max_images and max_images < len(img_ids):
        img_ids = img_ids[:max_images]
        print(f"[Eval] 限制: {len(img_ids)} 张")

    predictions:       Dict[int, str] = {}
    inject_log:        Dict[int, str] = {}
    vote_topk_log:     Dict[int, str] = {}
    vote_denoised_log: Dict[int, str] = {}
    implicit_log:      Dict[int, str] = {}
    per_query_log:     Dict[int, str] = {}
    roi_source_counter: Counter       = Counter()
    skipped = 0

    print(f"\n{'='*72}")
    print(f"开始推理 — {len(img_ids)} 张  |  每 {print_every} 步打印详细信息")
    print(f"  v6.2.1-YOLOE: LAVIS 视觉对齐 + per-query-token 检索 + 仅 NTP loss")
    print(f"  Token 布局: {_get_token_layout(engine.model)} (动态)")
    print(f"  投票阈值: VOTE_THRESH={_VOTE_THRESH}")
    print(f"  EVAL gen: beams={_EVAL_GEN_KWARGS['num_beams']} "
          f"max_new={_EVAL_GEN_KWARGS['max_new_tokens']} "
          f"min={_EVAL_GEN_KWARGS['min_new_tokens']} "
          f"rep={_EVAL_GEN_KWARGS['repetition_penalty']} "
          f"no_rep_ngram={_EVAL_GEN_KWARGS['no_repeat_ngram_size']}")
    print(f"  ROI 来源: YOLOe 在线检测 > grid (free setup, 无 GT 泄露)")
    if engine.yoloe_loader and engine.yoloe_loader.is_loaded:
        print(f"  YOLOe: conf={engine.yoloe_loader.conf}  "
              f"imgsz={engine.yoloe_loader.imgsz}  "
              f"max_rois={engine.yoloe_loader.max_rois}")
    else:
        print(f"  ⚠️  YOLOe 未加载, 全部走 grid fallback")
    print(f"{'='*72}\n")

    pbar = tqdm(enumerate(img_ids), total=len(img_ids), desc="Inference")
    for step, iid in pbar:
        fname    = id_to_file.get(iid, f"COCO_val2014_{int(iid):012d}.jpg")
        img_path = os.path.join(image_dir, fname)
        if not os.path.exists(img_path):
            img_path = os.path.join(image_dir, os.path.basename(fname))
        if not os.path.exists(img_path):
            skipped += 1; continue

        try:
            image     = Image.open(img_path).convert('RGB')
            do_detail = show_inject and ((step + 1) % print_every == 0)

            t0 = time.time()
            if do_detail:
                (caption, retrieval_str, vote_topk_str,
                 vote_denoised_str, implicit_str, roi_source,
                 per_query_diag_str) = \
                    engine.generate_single_with_detail(image, iid)
            else:
                caption = engine.generate_single(image, iid)
                retrieval_str = vote_topk_str = vote_denoised_str = ""
                implicit_str = per_query_diag_str = ""
                # 判断 ROI 来源 (从缓存或检测结果)
                if engine.yoloe_loader and engine.yoloe_loader.has_image(iid):
                    roi_source = 'yoloe'
                else:
                    roi_source = 'grid'

            elapsed_ms = (time.time() - t0) * 1000
            roi_source_counter[roi_source] += 1

            predictions[int(iid)] = caption
            if retrieval_str:      inject_log[int(iid)]        = retrieval_str
            if vote_topk_str:      vote_topk_log[int(iid)]     = vote_topk_str
            if vote_denoised_str:  vote_denoised_log[int(iid)] = vote_denoised_str
            if implicit_str:       implicit_log[int(iid)]      = implicit_str
            if per_query_diag_str: per_query_log[int(iid)]     = per_query_diag_str

            if do_detail:
                gt_caps = captions.get(iid, [])
                _print_step_detail(
                    step=step + 1, iid=iid, fname=fname,
                    caption=caption, gt_caps=gt_caps,
                    retrieval_str=retrieval_str,
                    vote_topk_str=vote_topk_str,
                    vote_denoised_str=vote_denoised_str,
                    implicit_str=implicit_str,
                    roi_source=roi_source,
                    elapsed_ms=elapsed_ms,
                    per_query_diag_str=per_query_diag_str,
                    vote_thresh=_VOTE_THRESH,
                )

        except Exception as e:
            tqdm.write(f"  ⚠️ [{iid}] 失败: {e}")
            skipped += 1

    print(f"\n[Eval] 完成: {len(predictions)} 条, {skipped} 跳过")
    print(f"  ROI 来源统计: YOLOe={roi_source_counter.get('yoloe', 0)} "
          f"grid={roi_source_counter.get('grid', 0)}")

    if not predictions:
        print("❌ 没有生成任何 caption"); return {}

    # ── 保存结果 ──────────────────────────────────────────────
    full_results = []
    for iid in img_ids:
        int_id = int(iid)
        if int_id not in predictions: continue
        full_results.append({
            'image_id':      int_id,
            'file_name':     id_to_file.get(iid, ''),
            'prediction':    predictions[int_id],
            'retrieval':     inject_log.get(int_id, ''),
            'vote_topk':     vote_topk_log.get(int_id, ''),
            'vote_denoised': vote_denoised_log.get(int_id, ''),
            'name_tokens':   implicit_log.get(int_id, ''),
            'per_query':     per_query_log.get(int_id, ''),
            'ground_truths': captions.get(iid, []),
        })

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(full_results, f, indent=2, ensure_ascii=False)
    print(f"[Eval] 结果保存: {output_file}")

    # ── 汇总样本 ──────────────────────────────────────────────
    print(f"\n{'='*72}")
    print(f"前 15 张预测样本 (v6.2.1-YOLOE | LAVIS 视觉对齐 | per-query 检索"
          f" | thresh={_VOTE_THRESH})")
    print(f"{'='*72}")
    roi_icons = {'yoloe': '🔍YOLOe', 'grid': '⬜grid'}
    shown = 0
    for iid in img_ids:
        int_id = int(iid)
        if int_id not in predictions: continue
        if engine.yoloe_loader and engine.yoloe_loader.has_image(iid):
            rsrc = 'yoloe'
        else:
            rsrc = 'grid'
        print(f"\n  [{iid}] {id_to_file.get(iid, '')}  {roi_icons.get(rsrc, rsrc)}")
        for gi, gt in enumerate(captions.get(iid, [])[:5], 1):
            print(f"    GT-{gi}: {_ensure_str(gt)}")
        if int_id in inject_log:
            print(f"    PKL检索:   [{inject_log[int_id][:100]}]")
        if int_id in vote_topk_log:
            print(f"    Step1:    {vote_topk_log[int_id][:100]}")
        if int_id in vote_denoised_log:
            print(f"    Step2去噪: {vote_denoised_log[int_id][:100]}")
        if int_id in implicit_log:
            print(f"    name_tok: [{implicit_log[int_id][:100]}]")
        print(f"    预测:  {predictions[int_id]}")
        print(f"  {'─'*68}")
        shown += 1
        if shown >= 15: break

    # ── COCO 指标 ─────────────────────────────────────────────
    print(f"\n[Eval] 计算 COCO 指标...")
    scores = compute_coco_metrics(predictions, captions)
    if scores:
        print(f"\n{'='*72}")
        print(f"COCO 2014 Karpathy Test  —  {len(predictions)} 张  (v6.2.1-YOLOE, free setup)")
        print(f"  ROI: YOLOe={roi_source_counter.get('yoloe', 0)} "
              f"grid={roi_source_counter.get('grid', 0)}")
        print(f"{'='*72}")
        for metric, val in sorted(scores.items()):
            print(f"  {metric:20s}: {val:.4f}")
        print(f"{'='*72}")

    return scores


# ================================================================
# single 模式
# ================================================================

def run_single(engine: InferenceEngine, image_path: str) -> str:
    print(f"\n[Single] 推理: {image_path}")
    image = Image.open(image_path).convert('RGB')
    t0 = time.time()
    (caption, retrieval_str, vote_topk_str,
     vote_denoised_str, implicit_str, roi_source,
     per_query_diag_str) = \
        engine.generate_single_with_detail(image, img_id=None)
    elapsed_ms = (time.time() - t0) * 1000

    roi_icons = {'yoloe': '🔍YOLOe-ROI', 'grid': '⬜grid-ROI'}
    print(f"\n{'─'*72}")
    print(f"  文件:          {os.path.basename(image_path)}")
    print(f"  耗时:          {elapsed_ms:.0f}ms")
    print(f"  ROI 来源:      {roi_icons.get(roi_source, roi_source)}")
    print(f"  PKL检索:       [{retrieval_str}]")
    print(f"  Step1:         {vote_topk_str}")
    print(f"  Step2 去噪后:  {vote_denoised_str}")
    print(f"  name_tokens:   [{implicit_str}]")
    if per_query_diag_str:
        print(f"  [PER-Q] per-query-token 检索:")
        for ln in per_query_diag_str.splitlines():
            print(f"    {ln}")
    print(f"  ✅ 预测:       {caption}")
    print(f"{'─'*72}")
    return caption


# ================================================================
# batch 模式
# ================================================================

def run_batch(engine: InferenceEngine, image_dir: str,
              output_file: str = 'batch_results.json',
              max_images: Optional[int] = None,
              print_every: int = 100) -> Dict[str, str]:
    exts  = ('*.jpg', '*.jpeg', '*.png', '*.bmp', '*.webp')
    paths = sorted({p for ext in exts
                    for p in glob.glob(os.path.join(image_dir, ext))
                             + glob.glob(os.path.join(image_dir, ext.upper()))})
    if max_images: paths = paths[:max_images]
    print(f"\n[Batch] {len(paths)} 张图片  print_every={print_every}")

    results: Dict[str, str] = {}
    roi_source_counter: Counter = Counter()

    for i, path in enumerate(tqdm(paths, desc="Batch")):
        try:
            fname  = os.path.basename(path)
            img_id = str(_extract_image_id_from_filename(fname) or '')
            do_detail = ((i + 1) % print_every == 0)

            if do_detail:
                (caption, retrieval_str, vote_topk_str,
                 vote_denoised_str, implicit_str, roi_source,
                 per_query_diag_str) = \
                    engine.generate_single_with_detail(
                        Image.open(path).convert('RGB'), img_id or None)
                roi_source_counter[roi_source] += 1
                _print_step_detail(
                    step=i + 1, iid=img_id, fname=fname,
                    caption=caption, gt_caps=[],
                    retrieval_str=retrieval_str,
                    vote_topk_str=vote_topk_str,
                    vote_denoised_str=vote_denoised_str,
                    implicit_str=implicit_str,
                    roi_source=roi_source,
                    elapsed_ms=0,
                    per_query_diag_str=per_query_diag_str,
                    vote_thresh=_VOTE_THRESH)
            else:
                caption = engine.generate_single(
                    Image.open(path).convert('RGB'), img_id or None)

            results[fname] = caption
        except Exception as e:
            tqdm.write(f"  ⚠️ [{os.path.basename(path)}] 失败: {e}")

    print(f"\n[Batch] ROI 来源: {dict(roi_source_counter)}")

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump([{'image': k, 'caption': v} for k, v in results.items()],
                  f, indent=2, ensure_ascii=False)
    print(f"\n[Batch] 保存: {output_file}")
    return results


# ================================================================
# main
# ================================================================

# ★ V6.2.1: ckpt 默认路径改成 train.py v6.2.1 实际写入位置
_DEFAULT_CKPT_PATH = (
    '/root/autodl-tmp/wh5319/silip_clip_caption/'
    'image_captioning_final/outputs/checkpoints/stage2_final.pt'
)

# ★ YOLOE 默认路径
_DEFAULT_YOLOE_PATH = (
    '/root/autodl-tmp/wh5319/silip_clip_caption/'
    'model/yoloe26_m_seg/model.pt'
)


def main():
    ap = argparse.ArgumentParser(
        description="Image Caption v6.2.1-YOLOE 推理 "
                    "(caption_model v6.4 + LAVIS 视觉对齐 + per-query 检索 + "
                    "YOLOe free setup, 无 GT 泄露)")
    ap.add_argument('--mode',        default='eval',
                    choices=['eval', 'single', 'batch'])
    ap.add_argument('--checkpoint',  default=_DEFAULT_CKPT_PATH)
    ap.add_argument('--ablation',    default='full')
    ap.add_argument('--device',      default='cuda')
    ap.add_argument('--print_every', type=int, default=100)

    # eval 相关
    ap.add_argument('--karpathy_json',
        default='/root/autodl-tmp/wh5319/all_data/coco2014_karpathy/'
                'coco_karpathy_test.json')
    ap.add_argument('--image_dir',
        default='/root/autodl-tmp/wh5319/all_data/coco_2014/val2014')
    ap.add_argument('--output_file', default='eval_results_v621_yoloe.json')
    ap.add_argument('--max_images',  type=int, default=5000)

    # [YOLOE-ROI] YOLOe 检测器参数
    ap.add_argument('--yoloe_path', default=_DEFAULT_YOLOE_PATH,
                    help='YOLOe 模型路径 (默认 yoloe26_m_seg)')
    ap.add_argument('--yoloe_conf', type=float, default=0.25,
                    help='YOLOe 置信度阈值 (默认 0.25)')
    ap.add_argument('--yoloe_imgsz', type=int, default=640,
                    help='YOLOe 输入图片尺寸 (默认 640)')
    ap.add_argument('--no_yoloe', action='store_true',
                    help='禁用 YOLOe, 全部用 grid fallback')

    # single
    ap.add_argument('--image_path', default=None)

    # BLIP2
    ap.add_argument('--blip2_path', default=None)

    # [v6.2.1] per-query 检索诊断开关
    ap.add_argument('--show_retrieval_diag', action='store_true', default=True,
                    help='详细打印 per-query-token 检索 (32q 各自的 top-1)')
    ap.add_argument('--no_retrieval_diag', action='store_false',
                    dest='show_retrieval_diag',
                    help='关闭 per-query-token 检索打印')

    args = ap.parse_args()

    config = get_ablation_config(args.ablation)
    if args.blip2_path:
        config.paths.blip2_qformer_path = args.blip2_path

    yoloe_path = None if args.no_yoloe else args.yoloe_path

    print(f"\n{'='*72}")
    print(f"Image Caption v6.2.1-YOLOE 推理  (caption_model v6.4 + YOLOe free setup)")
    print(f"  架构: LAVIS 视觉对齐 + per-query-token 检索 + V6.6 ROI 受限 attr + 仅 NTP loss")
    print(f"  [v6.2.1-IMP] from caption_model import (v6.4 包结构, 兼容老接口)")
    print(f"  [v6.2.1-VIS] 视觉编码器: LAVIS eva_vit_g.pth (BLIP-2 官方)")
    print(f"                Preprocess: Resize((224,224)) BICUBIC + CLIP mean/std")
    print(f"                → 与 ext_memory_lvis.pkl 构建路径完全一致")
    print(f"  [INIT-LAVIS] Q-Former cross-attn K/V ← LAVIS eva_vit_g.pth 最后层")
    print(f"  [PER-Q]      per-query-token 检索 (32q 独立 top-5 合并去重)")
    print(f"  [INIT-1]     CrossViT q/k/v ← BLIP2 self-attn")
    print(f"  [INIT-2]     MergeGate gate_mlp ← Normal(0,0.01)")
    print(f"  [INIT-3]     NameVisPrompt gate=−2.0, proj Normal(0,0.02)")
    print(f"  [INIT-4]     roi_retrieval_proj ← eye()")
    print(f"  Token 布局:  动态 '32vis + N流' (从 soft_retriever.streams 读取)")
    print(f"  Loss:        仅 NTP cross-entropy, label_smoothing=0.0")
    print(f"  [YOLOE] ROI: YOLOe > grid (无 GT 泄露, free / end-to-end setup)")
    print(f"  EVAL gen: max_new={_EVAL_GEN_KWARGS['max_new_tokens']}"
          f" min={_EVAL_GEN_KWARGS['min_new_tokens']}"
          f" beams={_EVAL_GEN_KWARGS['num_beams']}"
          f" rep={_EVAL_GEN_KWARGS['repetition_penalty']}"
          f" no_rep_ngram={_EVAL_GEN_KWARGS['no_repeat_ngram_size']}")
    print(f"  模式: {args.mode} | ckpt: {args.checkpoint}")
    print(f"  YOLOe:               {'禁用' if args.no_yoloe else yoloe_path}")
    if not args.no_yoloe:
        print(f"  YOLOe conf:          {args.yoloe_conf}")
        print(f"  YOLOe imgsz:         {args.yoloe_imgsz}")
    print(f"  show_retrieval_diag: {args.show_retrieval_diag}")
    print(f"{'='*72}\n")

    engine = InferenceEngine(
        config, args.checkpoint,
        device=args.device,
        yoloe_path=yoloe_path,
        yoloe_conf=args.yoloe_conf,
        yoloe_imgsz=args.yoloe_imgsz,
        show_retrieval_diag=args.show_retrieval_diag,
    )

    if args.mode == 'eval':
        run_eval(
            engine, args.image_dir, args.karpathy_json,
            args.output_file, args.max_images,
            print_every=args.print_every,
            show_inject=True,
        )
    elif args.mode == 'single':
        if not args.image_path:
            print("❌ single 模式需要 --image_path"); return
        run_single(engine, args.image_path)
    elif args.mode == 'batch':
        run_batch(engine, args.image_dir, args.output_file,
                  args.max_images, print_every=args.print_every)


if __name__ == "__main__":
    main()