#!/usr/bin/env python3
"""
retriever.py ── PKL Memory Retriever + Memory Bank 构建
[V6.9.3] v6.8.2 GPU 加速 + 新增 ROI 受限检索 (诊断实测 sim +0.245)

═══════════════════════════════════════════════════════════════════════
v6.9.3 新增 (相对 v6.8.2):
  ★ retrieve_attrs_per_roi_constrained 方法
    - 用 ROI 视觉特征作 query (替代 global 32q)
    - 只在该 ROI 物体类别的 attr 子集里检索 (绕过库内塌缩)
    - 自然解决 obj-attr 配对错位 (motorcycle 的 'red' 不会跑到 person 上)

  ★ COCO_TO_ATTR_ALIAS 同义词映射 + _resolve_obj_alias 方法
    - 解决 COCO 80 类里 17 个用别名的物体 (couch→sofa, donut→doughnut 等)
    - D1 覆盖率 78.8% → ~88-90% (救回 8/17 个误归类的物体)

  ★ 完全向后兼容
    - 旧方法 retrieve_*_per_query / retrieve_attrs / retrieve_name_topk 一字未改
    - 旧 ckpt / model.py / train.py / infer.py 零修改即可工作
    - 新方法只是"额外加的 API", 不替换任何旧路径
    - retrieve_attrs / retrieve_attrs_per_query 内部走 alias 解析,
      让旧路径也能受益于 couch→sofa 这种映射

诊断数据 (来自 diagnose_roi_attr_match.py 100 张图):
  方案 A (current global): mean sim 0.514
  方案 B (ROI + 全库):     mean sim 0.849 (+0.335) 但有配对错位
  方案 C (本版本 ROI 受限): mean sim 0.759 (+0.245) 配对正确

═══════════════════════════════════════════════════════════════════════
v6.8.2 改动 (保留, 无变化):
  1. __init__ 末尾 keys 常驻 GPU
  2. _load_*_pkl 已 normalize, 运行时不再重算
  3. retrieve_*_per_query 全程 GPU 计算
  4. 减少 .cpu().tolist() 同步, 用 GPU mask + bool indexing
  5. 诊断信息 (_last_per_query_top1) 保留, 仅 sample step 同步 CPU
"""

import os
import pickle
from typing import Optional, Dict, List
from collections import Counter

import torch
import torch.nn.functional as F
import numpy as np

from .constants import INVALID_TAG_NAMES_EXTENDED, NAME_SIM_THRESHOLD
from .utils import evcap_pre_name


# ════════════════════════════════════════════════════════════════════
# [V6.9.3 NEW] COCO 80 类 → attr_memory obj_name 同义词映射
# ════════════════════════════════════════════════════════════════════
# 诊断 D1 显示: COCO 80 类里 17 个不在 attr_memory 直接命中, 其中 8 个是
# 命名差异导致的 (attr_memory 用 sofa/doughnut/television 等 LVIS 标准名,
# COCO 用 couch/donut/tv 等口语名). 加同义词映射可以救回 8/17.
#
# 剩 9 个真没在 attr_memory 里 (bus/car/keyboard/laptop/microwave/mouse/
# orange/potted plant/train), 需要重建 attr_memory 才能覆盖, 此版本先不管.
# ════════════════════════════════════════════════════════════════════
COCO_TO_ATTR_ALIAS: Dict[str, str] = {
    'couch':         'sofa',
    'donut':         'doughnut',
    'tv':            'television',
    'cell phone':    'cellphone',
    'cellphone':     'cellphone',
    'hot dog':       'hotdog',
    'hotdog':        'hotdog',
    'fire hydrant':  'fireplug',
    'remote':        'remote control',
    'tie':           'necktie',
    # 以下是常见非 COCO80 但可能用到的别名
    'wine glass':    'wineglass',
    'sports ball':   'ball',
    'dining table':  'table',
    'hair drier':    'hair dryer',
}


# ============================================================
# PKLMemoryRetriever
# ============================================================
class PKLMemoryRetriever:
    """
    PKL 文件检索器 — [V6.9.3] keys 常驻 GPU + 新增 ROI 受限检索

    支持:
      - 名称库 (key + value 字符串)
      - 属性库 (key + value + dim + obj 映射)
      - per-query-token 独立检索 (Q个 query 各自检索 top-K, 合并去重)
      - [V6.9.3] ROI 受限检索 (per-ROI top-1 in obj's attr subset)
    """

    def __init__(self, name_pkl_path, attr_pkl_path,
                 attr_map_path=None, top_k_name=8,
                 max_attrs=4, name_sim_threshold=NAME_SIM_THRESHOLD,
                 device=None):
        self.top_k_name = top_k_name
        self.max_attrs = max_attrs
        self.name_sim_threshold = name_sim_threshold
        self._loaded = False
        self.name_keys: Optional[torch.Tensor] = None
        self.name_values: Optional[List[str]] = None
        self.attr_keys: Optional[torch.Tensor] = None
        self.attr_values: Optional[List[str]] = None
        self.attr_dims: Optional[List[str]] = None
        self.attr_objs: Optional[List[str]] = None
        self.name2attr_idx: Dict[str, List[int]] = {}
        self._name2idx: Dict[str, int] = {}

        if device is None:
            device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self._device = torch.device(device)
        self._invalid_names_set = set(INVALID_TAG_NAMES_EXTENDED)

        self._load_name_pkl(name_pkl_path)
        self._load_attr_pkl(attr_pkl_path, attr_map_path)

        if self._loaded:
            self._move_keys_to_device(self._device)
            print(f"  [PKLRetriever] ✅ 就绪 (V6.9.3 + ROI 受限检索)")
            print(f"    名称库: {self.name_keys.shape}  {len(self._name2idx)} 类  "
                  f"device={self.name_keys.device}")
            if self.attr_keys is not None:
                print(f"    属性库: {self.attr_keys.shape}  "
                      f"{len(self.name2attr_idx)} 物体有属性  "
                      f"device={self.attr_keys.device}")
                print(f"    [V6.9.3] COCO→attr 同义词: {len(COCO_TO_ATTR_ALIAS)} 条")

    def _move_keys_to_device(self, device):
        """[V6.8.2] 把 keys 搬到指定 device, 一次性"""
        device = torch.device(device)
        self._device = device
        if self.name_keys is not None:
            self.name_keys = self.name_keys.to(device, non_blocking=True).contiguous()
        if self.attr_keys is not None:
            self.attr_keys = self.attr_keys.to(device, non_blocking=True).contiguous()
        bytes_used = 0
        if self.name_keys is not None:
            bytes_used += self.name_keys.element_size() * self.name_keys.numel()
        if self.attr_keys is not None:
            bytes_used += self.attr_keys.element_size() * self.attr_keys.numel()
        mb = bytes_used / (1024 * 1024)
        print(f"  [PKLRetriever] keys 常驻 {device}, 占用 ~{mb:.1f} MB")

    def set_device(self, device):
        self._move_keys_to_device(device)

    def _load_name_pkl(self, path):
        if not os.path.exists(path):
            print(f"  [PKLRetriever] ⚠️  名称库不存在: {path}")
            return
        try:
            with open(path, 'rb') as f:
                nm = pickle.load(f)
            if isinstance(nm, dict):
                nk = nm.get('keys', nm.get('key'))
                nv = nm.get('values', nm.get('value'))
            elif isinstance(nm, (list, tuple)) and len(nm) == 2:
                nk, nv = nm
            else:
                raise ValueError(f"无法解析名称pkl格式: {type(nm)}")
            if isinstance(nk, np.ndarray):
                nk = torch.from_numpy(nk.astype(np.float32))
            nk = F.normalize(nk.float(), p=2, dim=-1).contiguous()
            if isinstance(nv, torch.Tensor):
                nv = nv.tolist()
            if isinstance(nv, np.ndarray):
                nv = nv.tolist()
            nv_clean = [evcap_pre_name(str(v)) for v in nv]
            self.name_keys = nk
            self.name_values = nv_clean
            for i, n in enumerate(nv_clean):
                if n and n not in self._name2idx:
                    self._name2idx[n] = i
            print(f"  [PKLRetriever] 名称库: {nk.shape}  "
                  f"{len(self._name2idx)} 类  dim={nk.shape[-1]}")
            self._loaded = True
        except Exception as e:
            print(f"  [PKLRetriever] ❌ 名称库加载失败: {e}")

    def _load_attr_pkl(self, attr_path, attr_map_path):
        if not os.path.exists(attr_path):
            print(f"  [PKLRetriever] ⚠️  属性库不存在: {attr_path}")
            return
        try:
            with open(attr_path, 'rb') as f:
                am = pickle.load(f)
            ak = am.get('keys')
            av = am.get('attr_values', am.get('values', []))
            ad = am.get('dims', [])
            ao = am.get('obj_names', [])
            n2a = am.get('name2attr_idx', {})
            if isinstance(ak, np.ndarray):
                ak = torch.from_numpy(ak.astype(np.float32))
            ak = F.normalize(ak.float(), p=2, dim=-1).contiguous()
            self.attr_keys = ak
            self.attr_values = [str(v) for v in av] if av else []
            self.attr_dims = list(ad)
            self.attr_objs = [str(o).lower().strip() for o in ao] if ao else []
            self.name2attr_idx = {
                evcap_pre_name(k): v for k, v in n2a.items()
            }
            unique_dims = sorted(set(self.attr_dims)) if self.attr_dims else []
            print(f"  [PKLRetriever] 属性库: {ak.shape}  "
                  f"{len(self.name2attr_idx)} 物体有属性")
            print(f"  [PKLRetriever] 属性维度数: {len(unique_dims)}")
        except Exception as e:
            print(f"  [PKLRetriever] ❌ 属性库加载失败: {e}")
            return
        if attr_map_path and os.path.exists(attr_map_path):
            try:
                import json
                with open(attr_map_path, encoding='utf-8') as f:
                    jmap = json.load(f)
                for name, info in jmap.items():
                    key = evcap_pre_name(name)
                    idxs = info.get('attr_indices', [])
                    if idxs:
                        self.name2attr_idx[key] = idxs
                print(f"  [PKLRetriever] name2attr_topk.json 补充: {len(jmap)} 条")
            except Exception as e:
                print(f"  [PKLRetriever] ⚠️  attr_map 加载失败: {e}")

    # ════════════════════════════════════════════════════════════════
    # [V6.9.3 NEW] 同义词解析 + ROI 受限检索 ★★★ 核心新方法 ★★★
    # ════════════════════════════════════════════════════════════════
    def _resolve_obj_alias(self, obj_name: str) -> str:
        """
        [V6.9.3] COCO 类名 → attr_memory 实际存储的 obj_name

        策略:
          1) 先用原始 name (经 evcap_pre_name 清洗) 查 name2attr_idx
          2) 没命中再查 COCO_TO_ATTR_ALIAS
          3) 还没命中返回原始 name (上层会自然走 fallback)

        例:
          'couch'  → 'sofa'      (attr_memory 里只有 sofa)
          'tv'     → 'television' (attr_memory 里只有 television)
          'person' → 'person'    (直接命中)
        """
        clean = evcap_pre_name(obj_name) if obj_name else ''
        if not clean:
            return ''
        if clean in self.name2attr_idx:
            return clean
        if clean in COCO_TO_ATTR_ALIAS:
            alias = COCO_TO_ATTR_ALIAS[clean]
            if alias in self.name2attr_idx:
                return alias
        return clean

    @torch.no_grad()
    def retrieve_attrs_per_roi_constrained(
            self,
            roi_features: torch.Tensor,         # [N_roi, D]
            roi_obj_names: List[str],            # [N_roi]
            *,
            threshold: float = 0.50,
            max_attrs_per_roi: int = 1,
            fallback_global: bool = False,
            fallback_threshold: float = 0.55,
            collect_diag: bool = False,
    ) -> List[Dict]:
        """
        [V6.9.3 NEW] ROI + 受限检索 (实测 sim +0.245 vs global)

        Args:
            roi_features:   ROI 视觉特征 [N_roi, D]
                            D 必须 == self.attr_keys.shape[-1] (768)
                            调用方应该传 retrieval_vecs[b] (per-ROI 32q 平均后的 768d)
            roi_obj_names:  每个 ROI 的物体名 (从 COCO category_id → name 来的)
                            自动经 _resolve_obj_alias 解析同义词
            threshold:      受限检索的 sim 阈值 (实测 mean 0.76, 用 0.50 比较稳)
            max_attrs_per_roi: 每个 ROI 最多返回几个 attr
            fallback_global:   该 ROI 物体不在 attr_memory 时, 是否退回全库
                               默认 False (避免引入"motorcycle 的 red 配给 person")
            fallback_threshold: fallback 用的阈值 (更严)
            collect_diag:   是否收集调试信息

        Returns:
            List[Dict], 每个含:
              {
                'roi_idx':    int,    # 来自第几个 ROI
                'attr_idx':   int,    # 全局 attr_keys 行号
                'attr_value': str,    # e.g. 'red'
                'dim':        str,    # e.g. 'color'
                'attr_key':   Tensor, # [D] 视觉原型, 给 model.py 当 init_q
                'sim':        float,
                'fallback':   bool,
                'obj_name':   str,    # 实际查表用的 (经过 alias 解析)
              }
        """
        if self.attr_keys is None or roi_features is None:
            return []

        if not isinstance(roi_features, torch.Tensor):
            roi_features = torch.as_tensor(roi_features)

        target_device = self.attr_keys.device
        if roi_features.device != target_device:
            roi_features = roi_features.to(target_device, non_blocking=True)
        roi_features = roi_features.float()

        if roi_features.dim() == 1:
            roi_features = roi_features.unsqueeze(0)
        elif roi_features.dim() == 3:
            assert roi_features.shape[0] == 1, \
                "retrieve_attrs_per_roi_constrained 只支持单张图 (B=1)"
            roi_features = roi_features.squeeze(0)

        D_expect = self.attr_keys.shape[-1]
        D_got = roi_features.shape[-1]
        assert D_got == D_expect, (
            f"ROI 特征维度 {D_got} 跟 attr_keys 维度 {D_expect} 不匹配. "
            f"调用前请先把 ROI 投影到 {D_expect}d (用 retrieval_qformer 等)"
        )

        roi_features = F.normalize(roi_features, p=2, dim=-1)

        N_roi = roi_features.shape[0]
        if len(roi_obj_names) < N_roi:
            # padding 空字符串 (后续会触发 fallback 或直接 skip)
            roi_obj_names = list(roi_obj_names) + [''] * (N_roi - len(roi_obj_names))

        results = []
        for i in range(N_roi):
            roi_feat = roi_features[i]
            obj_name = roi_obj_names[i] if i < len(roi_obj_names) else ''
            resolved_obj = self._resolve_obj_alias(obj_name)
            attr_indices = self.name2attr_idx.get(resolved_obj, [])

            if attr_indices:
                # ── 受限检索: 只在该 obj 的 attr 子集里找 ──
                idx_t = torch.as_tensor(attr_indices,
                                         device=target_device,
                                         dtype=torch.long)
                sub_keys = self.attr_keys.index_select(0, idx_t)
                sims = roi_feat.unsqueeze(0) @ sub_keys.T
                sims_flat = sims.squeeze(0)

                k = min(max_attrs_per_roi, sims_flat.shape[0])
                top_v, top_local = sims_flat.topk(k)
                top_v_list = top_v.cpu().tolist()
                top_local_list = top_local.cpu().tolist()

                for v, li in zip(top_v_list, top_local_list):
                    if v < threshold:
                        break
                    gi = attr_indices[li]
                    av_str = (self.attr_values[gi].strip()
                              if gi < len(self.attr_values) else '')
                    if not av_str or av_str.isdigit() or len(av_str) <= 1:
                        continue
                    dim_str = (self.attr_dims[gi]
                               if gi < len(self.attr_dims) else 'unknown')
                    results.append({
                        'roi_idx':    i,
                        'attr_idx':   gi,
                        'attr_value': av_str,
                        'dim':        dim_str,
                        'attr_key':   self.attr_keys[gi],
                        'sim':        float(v),
                        'fallback':   False,
                        'obj_name':   resolved_obj,
                    })

            elif fallback_global:
                # ── fallback 全库 (该 obj 不在 attr_memory) ──
                sims = roi_feat.unsqueeze(0) @ self.attr_keys.T
                top_v, top_i = sims.squeeze(0).topk(max_attrs_per_roi)
                for v, gi in zip(top_v.cpu().tolist(), top_i.cpu().tolist()):
                    if v < fallback_threshold:
                        break
                    av_str = (self.attr_values[gi].strip()
                              if gi < len(self.attr_values) else '')
                    if not av_str or av_str.isdigit() or len(av_str) <= 1:
                        continue
                    dim_str = (self.attr_dims[gi]
                               if gi < len(self.attr_dims) else 'unknown')
                    results.append({
                        'roi_idx':    i,
                        'attr_idx':   gi,
                        'attr_value': av_str,
                        'dim':        dim_str,
                        'attr_key':   self.attr_keys[gi],
                        'sim':        float(v),
                        'fallback':   True,
                        'obj_name':   resolved_obj,
                    })

        return results

    # ────────────────────────────────────────────────────────────
    # [V6.8.2] per-query-token 检索 (核心方法, 完全保留)
    # ────────────────────────────────────────────────────────────
    def retrieve_names_per_query(self, query_tokens, top_k_per_query=5,
                                 sub_top_k=32, dedupe=True, device=None,
                                 collect_diag=True):
        if collect_diag:
            self._last_per_query_top1 = []
        self._last_num_unique_names = 0
        self._last_num_thresh_pass = 0

        if self.name_keys is None:
            return [], []

        if query_tokens.dim() == 2:
            query_tokens = query_tokens.unsqueeze(0)
        B, Q, D = query_tokens.shape
        assert B == 1, "retrieve_names_per_query 只支持单张图"

        target_device = self.name_keys.device
        q_flat = query_tokens.reshape(-1, D).to(target_device, non_blocking=True).float()
        q_flat = F.normalize(q_flat, p=2, dim=-1)

        sims = q_flat @ self.name_keys.T
        N = self.name_keys.shape[0]
        k = min(top_k_per_query, N)
        top_sims, top_idx = sims.topk(k, dim=-1)

        all_sims = top_sims.reshape(-1)
        all_idx = top_idx.reshape(-1)
        sorted_sims, sort_order = all_sims.sort(descending=True)
        sorted_idx = all_idx[sort_order]

        thresh_mask = sorted_sims >= self.name_sim_threshold
        n_pass = int(thresh_mask.sum().item())

        if collect_diag:
            top_idx_diag = top_idx[:, 0].cpu().tolist()
            top_sims_diag = top_sims[:, 0].cpu().tolist()
            for qi in range(Q):
                idx0 = top_idx_diag[qi]
                name0 = (self.name_values[idx0]
                         if idx0 < len(self.name_values) else "")
                self._last_per_query_top1.append((qi, name0, float(top_sims_diag[qi])))

        if n_pass == 0:
            return [], []

        cap = min(n_pass, sub_top_k * 4)
        keep_sims_cpu = sorted_sims[thresh_mask][:cap].cpu().tolist()
        keep_idx_cpu = sorted_idx[thresh_mask][:cap].cpu().tolist()

        out_names = []
        out_sims = []
        seen = set()
        invalid = self._invalid_names_set
        nv = self.name_values
        nv_len = len(nv)
        for s, idx in zip(keep_sims_cpu, keep_idx_cpu):
            if idx >= nv_len:
                continue
            name = nv[idx]
            if not name or name in invalid:
                continue
            if dedupe and name in seen:
                continue
            seen.add(name)
            out_names.append(name)
            out_sims.append(float(s))
            if len(out_names) >= sub_top_k:
                break

        self._last_num_thresh_pass = n_pass
        self._last_num_unique_names = len(out_names)
        return out_names, out_sims

    def retrieve_attrs_per_query(self, query_tokens, obj_name,
                                 top_k_per_query=3, max_attrs=None,
                                 device=None):
        """[V6.8.2 GPU + V6.9.3 alias 解析] 每个 query 独立检索属性"""
        if self.attr_keys is None:
            return []
        # ★ V6.9.3: 走 alias 解析, 让旧路径也能受益 (couch→sofa)
        obj_lower = self._resolve_obj_alias(obj_name)
        candidates = self.name2attr_idx.get(obj_lower, [])
        if not candidates:
            return []

        if query_tokens.dim() == 2:
            query_tokens = query_tokens.unsqueeze(0)
        B, Q, D = query_tokens.shape
        assert B == 1, "retrieve_attrs_per_query 只支持单张图"

        target_device = self.attr_keys.device
        q_flat = query_tokens.reshape(-1, D).to(target_device, non_blocking=True).float()
        q_flat = F.normalize(q_flat, p=2, dim=-1)

        cand_idx_t = torch.as_tensor(candidates, device=target_device, dtype=torch.long)
        cand_keys = self.attr_keys.index_select(0, cand_idx_t)
        sims = q_flat @ cand_keys.T

        k = min(top_k_per_query, cand_keys.shape[0])
        top_sims, top_idx_local = sims.topk(k, dim=-1)
        all_sims = top_sims.reshape(-1)
        all_idx_local = top_idx_local.reshape(-1)
        sorted_sims, sort_order = all_sims.sort(descending=True)
        sorted_idx_local = all_idx_local[sort_order]

        thresh_mask = sorted_sims >= 0.10
        if not thresh_mask.any():
            return []
        max_a = max_attrs or self.max_attrs
        cap = min(int(thresh_mask.sum().item()), max_a * 4)
        keep_sims_cpu = sorted_sims[thresh_mask][:cap].cpu().tolist()
        keep_idx_cpu = sorted_idx_local[thresh_mask][:cap].cpu().tolist()

        ad = self.attr_dims
        av = self.attr_values
        ad_len = len(ad) if ad else 0
        av_len = len(av) if av else 0

        seen_dims = set()
        results = []
        for s, li in zip(keep_sims_cpu, keep_idx_cpu):
            gi = candidates[li]
            dim = ad[gi] if gi < ad_len else 'unknown'
            val = av[gi] if gi < av_len else ''
            val_stripped = val.strip() if val else ''
            if not val_stripped:
                continue
            if val_stripped.isdigit():
                continue
            if len(val_stripped) <= 1:
                continue
            seen_dims.add(dim)
            results.append({'dim': dim, 'value': val_stripped, 'sim': float(s)})
            if len(results) >= max_a:
                break
        return results

    # ────────────────────────────────────────────────────────────
    # [V6.8.2] 单 query 检索 (兼容旧接口)
    # ────────────────────────────────────────────────────────────
    def retrieve_name_topk(self, query_768, topk=None):
        if self.name_keys is None:
            return []
        topk = topk or self.top_k_name
        target_device = self.name_keys.device

        if isinstance(query_768, np.ndarray):
            q = torch.from_numpy(query_768).to(target_device, non_blocking=True).float()
        else:
            q = query_768.to(target_device, non_blocking=True).float()
        q = F.normalize(q, p=2, dim=-1)

        sims = self.name_keys @ q
        k_pool = min(topk * 4, sims.shape[0])
        top_sims, top_idx = sims.topk(k_pool)

        thresh_mask = top_sims >= self.name_sim_threshold
        if not thresh_mask.any():
            return []
        keep_sims = top_sims[thresh_mask].cpu().tolist()
        keep_idx = top_idx[thresh_mask].cpu().tolist()

        seen = set()
        invalid = self._invalid_names_set
        nv = self.name_values
        nv_len = len(nv)
        results = []
        for sim, idx in zip(keep_sims, keep_idx):
            if idx >= nv_len:
                continue
            name = nv[idx]
            if name and name not in seen and name not in invalid:
                seen.add(name)
                results.append((name, float(sim)))
                if len(results) >= topk:
                    break
        return results

    def retrieve_name(self, query_768):
        res = self.retrieve_name_topk(query_768, topk=1)
        return res[0] if res else ("", 0.0)

    def retrieve_attrs(self, query_768, obj_name):
        """[V6.8.2 GPU + V6.9.3 alias 解析] 兼容旧接口 (numpy 入口)"""
        if self.attr_keys is None:
            return []
        # ★ V6.9.3: 走 alias 解析
        obj_lower = self._resolve_obj_alias(obj_name)
        candidates = self.name2attr_idx.get(obj_lower, [])
        if not candidates:
            return []
        target_device = self.attr_keys.device

        if isinstance(query_768, np.ndarray):
            q = torch.from_numpy(query_768).to(target_device, non_blocking=True).float()
        else:
            q = query_768.to(target_device, non_blocking=True).float()
        q = F.normalize(q, p=2, dim=-1)

        cand_idx_t = torch.as_tensor(candidates, device=target_device, dtype=torch.long)
        cand_keys = self.attr_keys.index_select(0, cand_idx_t)
        sims = cand_keys @ q

        k_pool = min(self.max_attrs * 3, sims.shape[0])
        top_sims, top_local = sims.topk(k_pool)
        thresh_mask = top_sims >= 0.10
        if not thresh_mask.any():
            return []
        keep_sims = top_sims[thresh_mask].cpu().tolist()
        keep_local = top_local[thresh_mask].cpu().tolist()

        ad = self.attr_dims
        av = self.attr_values
        ad_len = len(ad) if ad else 0
        av_len = len(av) if av else 0

        seen_dims = {}
        results = []
        for sim, li in zip(keep_sims, keep_local):
            gi = candidates[li]
            dim = ad[gi] if gi < ad_len else 'unknown'
            val = av[gi] if gi < av_len else ''
            if not val:
                continue
            val_stripped = val.strip()
            if not val_stripped:
                continue
            if val_stripped.isdigit():
                continue
            if len(val_stripped) <= 1:
                continue
            if dim not in seen_dims:
                seen_dims[dim] = True
                results.append({'dim': dim, 'value': val_stripped, 'sim': float(sim)})
            if len(results) >= self.max_attrs:
                break
        return results

    def get_name_pkl_indices(self, obj_name):
        obj_lower = evcap_pre_name(obj_name)
        if self.name_values is None:
            return []
        return [i for i, n in enumerate(self.name_values) if n == obj_lower]

    def get_all_attr_dims(self) -> List[str]:
        if not self.attr_dims:
            return []
        dim_counter = Counter(self.attr_dims)
        return [dim for dim, _ in dim_counter.most_common()]

    @property
    def name_dim(self):
        return self.name_keys.shape[-1] if self.name_keys is not None else 768

    @property
    def attr_dim(self):
        return self.attr_keys.shape[-1] if self.attr_keys is not None else 768


# ============================================================
# Memory Bank 构造 (保留, soft_retriever 已弃用但函数还在)
# ============================================================
def build_memory_bank_from_pkl(pkl_retriever: PKLMemoryRetriever, device) -> Dict:
    """[V6.8.1] soft_retriever 已删, 此函数仅保留兼容性, 实际不再被调用"""
    bank = {}
    if pkl_retriever.name_keys is not None:
        name_keys = pkl_retriever.name_keys.to(device)
        bank['name'] = {
            'keys':       name_keys,
            'val_embeds': name_keys.clone(),
            'texts':      pkl_retriever.name_values,
        }
    if (pkl_retriever.attr_keys is not None
            and pkl_retriever.attr_dims is not None):
        all_dims = pkl_retriever.get_all_attr_dims()
        for dim in all_dims:
            indices = [i for i, d in enumerate(pkl_retriever.attr_dims) if d == dim]
            if not indices:
                continue
            dim_keys = pkl_retriever.attr_keys[indices].to(device)
            bank[dim] = {
                'keys':       dim_keys,
                'val_embeds': dim_keys.clone(),
                'texts':      [pkl_retriever.attr_values[i] for i in indices],
            }
            print(f"  [MemBank] {dim}: {len(indices)} 条属性")
    print(f"  [MemBank] 总流数: {len(bank)}  (name + {len(bank)-1} 属性维度)")
    return bank