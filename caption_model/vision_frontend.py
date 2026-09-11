#!/usr/bin/env python3
"""
vision_frontend.py ── OnlineVisionFrontend  [SIGLIP-LEGACY-RENAMED]

═══════════════════════════════════════════════════════════════════
本次重命名 (相对历史版本):
  历史代码用 siglip_* / clip_* 命名实际承载的是 EVA-ViT-g 特征,
  没有任何真正的 SigLIP 编码器被加载. 本次彻底重命名:

  旧名                       新名                  含义
  ────────────────────────────────────────────────────────────────
  eva01_g / clip_global      eva_global            EVA 全局 raw [B,1408]
  siglip_global              eva_global_768        EVA 投影到 768d
  eva01_patches              eva_patches           EVA 末层 patch
  clip_patches / siglip_patch (合并)                ↑ 同上
  siglip_mid_patch           eva_mid_patches       EVA 倒数第 9 层
  eva01_rois / siglip_rois   eva_rois              ROI Align 输出

  保留为死字段 (按需求"不要删除", 仅改名):
  siglip_rois_for_retrieval  eva_global_per_roi    全局向量复制 N 份
  siglip_patch_projected     eva_global_per_roi_alt 同上
  siglip_roi_coord_patches   eva_roi_coord_patches  始终为 None
  clip_patch                 eva_patch_single       始终为 None
  clip_rois                  eva_rois_alt           始终为 None
═══════════════════════════════════════════════════════════════════

在线视觉前端,负责:
  - 加载 LAVIS BLIP-2 官方视觉编码器 eva_vit_g.pth
  - 用 forward hook 抓取中间层特征 (mid layer = depth - 9)
  - 从 patch features 用双线性 ROI Align 抽出 ROI 特征 (fallback)
  - 调用 ROICrossAttnRefinement 精炼 ROI
  - 计算空间编码 (abs_pos, rel_pos, is_core, importance, ...)
  - 提供 _simple_grid_boxes 作为无外部 box 时的兜底
"""

import os
import math as _m

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .constants import EVA01_DIM, QFORMER_HIDDEN, COCO_ID_TO_NAME, INVALID_TAG_NAMES_EXTENDED
from .fusion import ROICrossAttnRefinement


class OnlineVisionFrontend(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.eva_grid = 16
        self.eva_num_patches = 256
        self.eva_embed_dim = EVA01_DIM
        self.max_rois = getattr(config.preprocess, 'max_rois', 13)
        self.core_count = getattr(config.preprocess, 'core_count', 4)
        self._eva_model = None
        self._eva_proc = None

        # EVA 1408d → 768d 的投影 (历史误名为 siglip_global, 实际是这层的输出)
        self.global_proj = nn.Linear(EVA01_DIM, QFORMER_HIDDEN, bias=False)
        nn.init.orthogonal_(self.global_proj.weight)

        self.roi_refiner = ROICrossAttnRefinement(
            embed_dim=EVA01_DIM, num_heads=8, use_spatial_bias=True)
        self._loaded = False

    def load_encoders(self, device):
        if self._loaded:
            return
        self.global_proj = self.global_proj.to(device)
        self.roi_refiner = self.roi_refiner.to(device)

        eva_path = getattr(self.config.paths, 'eva_vit_g_model', None)
        if not eva_path or not os.path.exists(eva_path):
            eva_path = getattr(
                self.config.paths, 'eva01_model',
                '/root/autodl-tmp/wh5319/silip_clip_caption/image_captioning_final/eva_vit_g.pth')

        print(f"\n  [EVA-VIT-G] 加载 LAVIS BLIP-2 官方视觉编码器")
        print(f"  [EVA-VIT-G] 路径: {eva_path}")

        if not os.path.exists(eva_path):
            print(f"  [EVA-VIT-G] ❌ 权重文件不存在: {eva_path}")
            print(f"  [EVA-VIT-G] 将使用零特征作为 fallback")
            self._loaded = True
            return

        try:
            try:
                from lavis_eva_vit import VisionTransformer, interpolate_pos_embed
                from functools import partial
                print(f"  [EVA-VIT-G] 使用 lavis_eva_vit.VisionTransformer (内嵌版)")
            except ImportError as e:
                raise RuntimeError(f"导入 lavis_eva_vit 失败: {e}")

            print(f"  [EVA-VIT-G] 创建 VisionTransformer (embed_dim=1408, depth=39, heads=16)...")
            vit_model = VisionTransformer(
                img_size=224, patch_size=14, use_mean_pooling=False,
                embed_dim=1408, depth=39, num_heads=1408 // 88,
                mlp_ratio=4.3637, qkv_bias=True, drop_path_rate=0.0,
                norm_layer=partial(nn.LayerNorm, eps=1e-6), use_checkpoint=False)

            print(f"  [EVA-VIT-G] 加载 checkpoint (CPU)...")
            state_dict = torch.load(eva_path, map_location='cpu', weights_only=False)
            if isinstance(state_dict, dict) and 'model' in state_dict:
                state_dict = state_dict['model']
            try:
                interpolate_pos_embed(vit_model, state_dict)
            except Exception as e:
                print(f"  [EVA-VIT-G] interpolate_pos_embed 跳过: {e}")

            miss, unexp = vit_model.load_state_dict(state_dict, strict=False)
            print(f"  [EVA-VIT-G] load_state_dict: missing={len(miss)}, unexpected={len(unexp)}")
            if len(miss) == 0:
                print(f"  [EVA-VIT-G] ✅ 所有参数加载成功")
            else:
                print(f"  [EVA-VIT-G] ⚠️  missing 前5: {miss[:5]}")

            sample_key = 'blocks.0.norm1.weight'
            if sample_key in state_dict:
                orig_mean = state_dict[sample_key].float().mean().item()
                try:
                    loaded_mean = vit_model.blocks[0].norm1.weight.float().mean().item()
                    if abs(orig_mean - loaded_mean) < 1e-5:
                        print(f"  [EVA-VIT-G] ✅ 权重验证通过 (非随机初始化)")
                    else:
                        print(f"  [EVA-VIT-G] ⚠️  权重验证异常: orig={orig_mean:.4f} loaded={loaded_mean:.4f}")
                except Exception:
                    pass

            print(f"  [EVA-VIT-G] 移到 {device} (bf16)...")
            vit_model = vit_model.to(device=device, dtype=torch.bfloat16).eval()
            for p in vit_model.parameters():
                p.requires_grad = False

            from torchvision import transforms
            from torchvision.transforms.functional import InterpolationMode
            preprocess = transforms.Compose([
                transforms.Resize((224, 224), interpolation=InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.48145466, 0.4578275, 0.40821073),
                    std=(0.26862954, 0.26130258, 0.27577711)),
            ])

            self._eva_model = vit_model
            self._eva_proc = preprocess
            print(f"  [EVA-VIT-G] preprocess: Resize((224,224)) BICUBIC + CLIP mean/std ✅")
            print(f"  [EVA-VIT-G] ✅ LAVIS eva_vit_g 加载完成")

        except Exception as e:
            print(f"  [EVA-VIT-G] ❌ 加载失败: {e}")
            import traceback
            traceback.print_exc()
            print(f"  [EVA-VIT-G] 将使用零特征作为 fallback")

        self._loaded = True

    @torch.no_grad()
    def _encode_eva(self, images, device):
        """跑 EVA-ViT-g, 返回 (eva_global_raw, eva_patches, eva_mid_patches)"""
        if self._eva_model is None:
            B = len(images) if isinstance(images, list) else images.shape[0]
            return (torch.zeros(B, EVA01_DIM, device=device),
                    torch.zeros(B, 256, EVA01_DIM, device=device),
                    torch.zeros(B, 256, EVA01_DIM, device=device))

        if isinstance(images, list):
            inp = torch.stack([self._eva_proc(im) for im in images]).to(device)
        else:
            inp = images.to(device)
        B = inp.shape[0]
        expected = 256

        vit_model = self._eva_model
        blocks = vit_model.blocks
        mid_store = [None]
        hooks = []
        n_blk = len(blocks)
        mid_idx = max(0, n_blk - 9)

        def _make_hook(store):
            def fn(m, i, o):
                feat = o[0] if isinstance(o, tuple) else o
                store[0] = feat.detach().float()
            return fn

        hooks.append(blocks[mid_idx].register_forward_hook(_make_hook(mid_store)))

        try:
            _amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            with torch.amp.autocast(device_type="cuda", dtype=_amp_dtype):
                last_feat = vit_model(inp)
        finally:
            for h in hooks:
                h.remove()

        def _extract(feat):
            if feat is None:
                return (torch.zeros(B, EVA01_DIM, device=device),
                        torch.zeros(B, expected, EVA01_DIM, device=device))
            feat = feat.float()
            has_cls = (feat.shape[1] == expected + 1)
            g = feat[:, 0] if has_cls else feat.mean(1)
            p = feat[:, 1:] if has_cls else feat
            if p.shape[1] != expected:
                p = F.interpolate(
                    p.permute(0, 2, 1), size=expected,
                    mode='linear', align_corners=False).permute(0, 2, 1)
            return g.contiguous(), p.contiguous()

        g_raw, patches = _extract(last_feat)
        _, mid_patch = _extract(mid_store[0])
        g = F.normalize(g_raw, p=2, dim=-1)
        return g.float(), patches.float(), mid_patch.float()

    def _roi_from_patches_fallback(self, patches, global_feat, boxes_batch, grid_size):
        B, _, D = patches.shape
        H = W = grid_size
        roi_feats = torch.zeros(B, self.max_rois, D, device=patches.device, dtype=patches.dtype)
        for b in range(B):
            bxs = boxes_batch[b][:self.max_rois]
            p = patches[b]
            if p.shape[0] == H * W + 1:
                p = p[1:]
            elif p.shape[0] != H * W:
                for i in range(min(len(bxs), self.max_rois)):
                    roi_feats[b, i] = global_feat[b]
                continue
            grid = p.reshape(H, W, D)
            for i, box in enumerate(bxs):
                if i >= self.max_rois:
                    break
                x1, y1, x2, y2 = box['bbox']
                gx1 = int(max(0, x1 * W))
                gy1 = int(max(0, y1 * H))
                gx2 = min(W, int(x2 * W) + 1)
                gy2 = min(H, int(y2 * H) + 1)
                gx2 = max(gx2, gx1 + 1)
                gy2 = max(gy2, gy1 + 1)
                roi = grid[gy1:gy2, gx1:gx2, :]
                roi_feats[b, i] = roi.reshape(-1, D).mean(0) if roi.numel() > 0 else global_feat[b]
        if hasattr(self, 'roi_refiner') and self.roi_refiner is not None:
            try:
                p_seq = patches
                if p_seq.shape[1] == H * W + 1:
                    p_seq = p_seq[:, 1:]
                roi_feats = self.roi_refiner(
                    roi_feats.float(), p_seq.float(),
                    boxes_batch=boxes_batch, grid_size=grid_size).to(patches.dtype)
            except Exception:
                pass
        return roi_feats

    def _compute_spatial(self, boxes_batch):
        """计算 abs / rel 空间向量 (ViP-CAP 风格, 与 SpatialTagAnchor 配对)"""
        B = len(boxes_batch)
        M = self.max_rois
        abs_p = np.zeros((B, M, 10), dtype=np.float32)
        rel_p = np.zeros((B, M, M, 12), dtype=np.float32)
        is_c = np.zeros((B, M), dtype=bool)
        imp = np.zeros((B, M), dtype=np.float32)
        tag_conf = np.zeros((B, M), dtype=np.float32)
        cats = []
        tags = []
        msk = np.zeros((B, M), dtype=bool)
        for b in range(B):
            bxs = boxes_batch[b][:M]
            n = len(bxs)
            msk[b, :n] = True
            for i in range(n):
                x1, y1, x2, y2 = bxs[i]['bbox']
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                w, h = max(x2 - x1, 1e-6), max(y2 - y1, 1e-6)
                area = w * h
                abs_p[b, i] = [
                    cx, cy, w, h, area,
                    int(min(cx * 2.99, 2)) + int(min(cy * 2.99, 2)) * 3,
                    _m.sqrt((cx - .5) ** 2 + (cy - .5) ** 2),
                    w / max(h, 1e-6),
                    bxs[i].get('confidence', .5),
                    1.0 if area > .1 else 0.0,
                ]
                is_c[b, i] = bxs[i].get('is_core', False)
                imp[b, i] = bxs[i].get('confidence', .5)
                cid = bxs[i].get('category_id', 0)
                raw_name = bxs[i].get('name', '').strip().lower()
                if int(cid) in COCO_ID_TO_NAME:
                    tag_conf[b, i] = 1.0
                elif raw_name and raw_name not in INVALID_TAG_NAMES_EXTENDED:
                    tag_conf[b, i] = max(0.5, bxs[i].get('confidence', 0.5))
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    ci, cj = abs_p[b, i], abs_p[b, j]
                    dx, dy = cj[0] - ci[0], cj[1] - ci[1]
                    d = _m.sqrt(dx ** 2 + dy ** 2) if (dx or dy) else 0
                    a = _m.atan2(dy, dx) if d > 0 else 0
                    sl = _m.log(max(cj[4], 1e-8) / max(ci[4], 1e-8))
                    ix1 = max(ci[0] - ci[2] / 2, cj[0] - cj[2] / 2)
                    iy1 = max(ci[1] - ci[3] / 2, cj[1] - cj[3] / 2)
                    ix2 = min(ci[0] + ci[2] / 2, cj[0] + cj[2] / 2)
                    iy2 = min(ci[1] + ci[3] / 2, cj[1] + cj[3] / 2)
                    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
                    union = ci[4] + cj[4] - inter
                    iou = inter / max(union, 1e-8)
                    xo = max(0, min(ci[0] + ci[2] / 2, cj[0] + cj[2] / 2) -
                             max(ci[0] - ci[2] / 2, cj[0] - cj[2] / 2))
                    yo = max(0, min(ci[1] + ci[3] / 2, cj[1] + cj[3] / 2) -
                             max(ci[1] - ci[3] / 2, cj[1] - cj[3] / 2))
                    rel_p[b, i, j] = [
                        dx, dy, d, _m.sin(a), _m.cos(a),
                        sl, iou, d, xo, yo, j, abs(ci[8] - cj[8]),
                    ]
            cats.append([bx.get('category_id', 0) for bx in bxs] + [0] * (M - n))
            tags.append([bx.get('name', 'object') for bx in bxs] + ['object'] * (M - n))
        return abs_p, rel_p, is_c, imp, cats, tags, msk, tag_conf

    def _simple_grid_boxes(self, images):
        all_boxes = []
        for img in images:
            boxes = [{
                'bbox': [0.05, 0.05, 0.95, 0.95],
                'category_id': 0, 'name': 'scene',
                'confidence': 1.0, 'is_core': True,
            }]
            for r in range(3):
                for c in range(3):
                    boxes.append({
                        'bbox': [c / 3 + 0.02, r / 3 + 0.02,
                                 (c + 1) / 3 - 0.02, (r + 1) / 3 - 0.02],
                        'category_id': 0, 'name': 'region',
                        'confidence': 0.5, 'is_core': False,
                    })
            all_boxes.append(boxes[:self.max_rois])
        return all_boxes

    def forward(self, images, device, external_boxes=None):
        B = len(images)
        if external_boxes is not None:
            boxes_batch = []
            for b in range(B):
                if b < len(external_boxes) and external_boxes[b] is not None:
                    bxs = external_boxes[b][:self.max_rois]
                else:
                    bxs = [{
                        'bbox': [0.05, 0.05, 0.95, 0.95],
                        'category_id': 0, 'name': 'scene',
                        'confidence': 1.0, 'is_core': True,
                    }]
                for i, box in enumerate(bxs):
                    if 'is_core' not in box:
                        box['is_core'] = (i < self.core_count)
                    if 'name' not in box:
                        box['name'] = box.get('category_name', 'object')
                boxes_batch.append(bxs)
        else:
            boxes_batch = self._simple_grid_boxes(images)

        # ── EVA-ViT-g 推理 ──
        eva_global, eva_patches, eva_mid_patches = self._encode_eva(images, device)
        eva_rois = self._roi_from_patches_fallback(
            eva_patches, eva_global, boxes_batch, self.eva_grid)
        with torch.no_grad():
            eva_global_768 = F.normalize(
                self.global_proj(eva_global.float()), p=2, dim=-1)
        eva_global_per_roi = eva_global_768.unsqueeze(1).expand(
            B, self.max_rois, -1).contiguous()

        # ── 空间编码 ──
        ap, rp, ic, im, ca, ta, mk, tag_conf = self._compute_spatial(boxes_batch)

        return {
            # ─── EVA 主特征 (用得到) ─────────────────────────────
            'eva_global':              eva_global,            # [B, 1408]  raw
            'eva_global_768':          eva_global_768,        # [B, 768]   投影
            'eva_patches':             eva_patches,           # [B, 256, 1408]
            'eva_mid_patches':         eva_mid_patches,       # [B, 256, 1408]
            'eva_rois':                eva_rois,              # [B, N, 1408]

            # ─── 死字段 (保留, 仅改名) ───────────────────────────
            'eva_global_per_roi':      eva_global_per_roi,    # 全局复制 N 份
            'eva_global_per_roi_alt':  eva_global_per_roi,    # 同上 (老 siglip_patch_projected)
            'eva_roi_coord_patches':   None,                  # 始终 None
            'eva_patch_single':        None,                  # 始终 None
            'eva_rois_alt':            None,                  # 始终 None

            # ─── 空间相关 (ViP-CAP 风格 + SpatialTagAnchor) ──────
            'absolute_pos':            torch.from_numpy(ap).float().to(device),
            'relative_pos':            torch.from_numpy(rp).float().to(device),
            'is_core':                 torch.from_numpy(ic).to(device),
            'importance_scores':       torch.from_numpy(im).float().to(device),
            'category_ids':            ca,
            'tag_sequences':           ta,
            'roi_mask':                torch.from_numpy(mk).to(device),
            'attribute_features':      torch.zeros(B, self.max_rois, EVA01_DIM, device=device),
            'tag_confidence':          torch.from_numpy(tag_conf).float().to(device),
            'boxes_batch':             boxes_batch,
        }