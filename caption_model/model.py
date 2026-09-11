#!/usr/bin/env python3
"""
model.py v6.6.1 ── 修 v6.6 的 ROI feature 来源 bug

═══════════════════════════════════════════════════════════════════════
[V6.6.1 改动] (相对 v6.6):
  ★ Bug 修复: _collect_name_attr_tokens_evcap 里 ROI feature 用错了来源

  v6.6 (有 bug): roi_feats_b = retrieval_vecs[b]
                  ↑ retrieval_vecs 实际是 [B, 32, 768] 全图 query (不是 per-ROI)
                  → ROI feature 跟 obj_name 错位 → sim 低 → 大部分被 0.50 阈值过滤
                  → 退到 fallback 走 v6.5 老路径 (训练日志里 100% [FALLBACK])

  v6.6.1 (修对): roi_feats_b = roi_q_mean[b]
                  ↑ roi_q_mean 是 PerROIQFormer 输出 mean [B, N_roi, 768]
                  → 真正的 per-ROI 768d 特征, 跟 obj_name 顺序对齐

  ★ 同时把 threshold 从 0.50 → 0.30
    诊断里方案 C mean sim 是 0.759, 但那是用 retrieval_qformer 在 ROI crop 上
    跑出来的 (跟 attr_keys 同源 frozen evcap_retrieval_qformer);
    训练里 roi_q_mean 是用 evcap_qformer (训练中) 处理的, 不同 q-former,
    sim 会低一些 (估计 0.4-0.6), 0.50 太严, 大部分过不去.
    改 0.30 让 ROI 路径更多命中, 即使 sim 略低也比走 fallback (基于错的 name) 好.

  ★ ROI 诊断字段加了 'src' 标记, 方便排查到底走哪条路径

[V6.6 (前一版)] 改动:
  ★ _collect_name_attr_tokens_evcap Step 3 改 ROI 受限检索 (但用错了 query)

═══════════════════════════════════════════════════════════════════════
[V6.5 关键路径] (保留):
  1. attr_qformer = evcap_qformer 别名 (CIDEr -3 修复)
  2. init_q [name_8 | attr_8] 拼成 16 个 token, 一次过 evcap_qformer
  3. _legacy_prompt_v62=True 默认 (无 marker, name+attr 直连)
"""

import os

import torch
import torch.nn as nn
import torch.nn.functional as F

# 外部依赖 (不在本包内)
from spatial_encoding import SpatialEncoder

from .constants import (
    EVA01_DIM, QFORMER_HIDDEN, LLM_HIDDEN,
    INVALID_TAG_NAMES, INVALID_TAG_NAMES_EXTENDED, COCO_ID_TO_NAME,
    num_prefix, pluralize,
)
from .utils import clean_caption, evcap_pre_name
from .retriever import PKLMemoryRetriever
from .roi_pipeline import (
    ROIAlignPooler, ROINameAttrInjector, PerROIQFormer,
    ROITokenAggregator, ExplicitROIDeduplicator,
)
from .fusion import (
    AttrFinalGating, TriStreamFusion, GlobalROIBidirectionalFusion,
)
from .qformer import BLIP2QFormer
from .retrieval_qformer import RetrievalQFormer, AttrQFormer
from .auxiliary import (
    NameAttrDeduplicator, SpatialTokenEmbedder, EntityTextEmbedder,
    VisFeatureEnhancer, AoAGate, GlobalContextFusion,
    CollectiveObjectMerger, TypeEmbedding, CoreObjectAttention,
    SpatialTagAnchor, LlipMixtureTokens, ObjectTagEmbedder,
    ContrastiveHead,
)
from .losses import MultiPositiveSigmoidContrastLoss, ROIAttributeContrastLoss
from .decoder import MistralDecoder
from .vision_frontend import OnlineVisionFrontend


# ============================================================
# CaptionModel v6.6.1
# ============================================================
class CaptionModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_dim = EVA01_DIM

        # ── 视觉对齐 / 空间锚定 ──
        self.vis_enhancer = VisFeatureEnhancer(EVA01_DIM)
        self.spatial_tag_anchor = SpatialTagAnchor(
            embed_dim=EVA01_DIM,
            abs_dim=config.spatial.absolute_dim,
            rel_dim=config.spatial.relative_dim)
        self.type_emb = TypeEmbedding(8, EVA01_DIM)
        self.pos_emb = nn.Embedding(64, EVA01_DIM)
        self.global_ctx_fusion = GlobalContextFusion(EVA01_DIM)
        self.collective_merger = CollectiveObjectMerger(EVA01_DIM, merge_threshold=3)
        self.core_attention = CoreObjectAttention(EVA01_DIM)
        self.spatial_enc = SpatialEncoder(
            config.spatial.absolute_dim,
            config.spatial.relative_dim,
            config.spatial.hidden_dim,
            EVA01_DIM)
        self.explicit_roi_dedup = ExplicitROIDeduplicator(iou_threshold=0.7, max_rois=13)
        self.name_attr_dedup = NameAttrDeduplicator(
            qformer_dim=QFORMER_HIDDEN,
            name_sem_thresh=0.92, attr_sim_thresh=0.15,
            max_names=8, max_attrs_per_obj=3)
        self.global_roi_fusion = GlobalROIBidirectionalFusion(
            embed_dim=EVA01_DIM, num_heads=8, max_core=4, max_non_core=9)

        # ── 4 个 image Q-Former + 1 个 attribution Q-Former ──
        self.aux_qformer = BLIP2QFormer(
            num_queries=32, num_layers=12, num_heads=12, intermediate_size=3072)
        self.final_qformer = BLIP2QFormer(
            num_queries=32, num_layers=12, num_heads=12, intermediate_size=3072)
        self.evcap_qformer = BLIP2QFormer(
            num_queries=16, num_layers=12, num_heads=12, intermediate_size=3072)
        self.attr_qformer = self.evcap_qformer
        self.evcap_retrieval_qformer = BLIP2QFormer(
            num_queries=32, num_layers=12, num_heads=12, intermediate_size=3072)
        for p in self.evcap_retrieval_qformer.parameters():
            p.requires_grad = False
        self.retrieval_qformer = RetrievalQFormer(num_queries=32)

        # ── 融合 / 投影 ──
        self.aoa_gate = AoAGate(embed_dim=QFORMER_HIDDEN)
        self.tri_stream_fusion = TriStreamFusion(embed_dim=QFORMER_HIDDEN, num_heads=8)
        self.attr_final_gating = AttrFinalGating(embed_dim=QFORMER_HIDDEN, num_heads=8)
        self.contrastive_head = ContrastiveHead(vis_dim=QFORMER_HIDDEN, llm_hidden=LLM_HIDDEN)

        # ── ROI 管线 ──
        self.roi_align_pooler = ROIAlignPooler(roi_size=7, grid=16)
        self.roi_name_injector = ROINameAttrInjector(embed_dim=EVA01_DIM)
        self.per_roi_qformer = PerROIQFormer(qformer=self.evcap_qformer)
        self.roi_token_aggregator = ROITokenAggregator(
            embed_dim=QFORMER_HIDDEN, num_heads=8, num_queries=32)

        # ── 诊断状态 ──
        self._last_implicit_names = None
        self._last_implicit_valid_mask = None
        self._last_vote_topk = None
        self._last_vote_denoised = None
        self._last_attr_pieces = None
        self._last_roi_attr_diag = None

        # ── 可学习参数 ──
        self.roi_core_boost = nn.Parameter(torch.tensor(1.5))
        self.abs_pos_proj = nn.Sequential(
            nn.Linear(config.spatial.absolute_dim, EVA01_DIM),
            nn.LayerNorm(EVA01_DIM))
        self.rel_pos_proj = nn.Sequential(
            nn.Linear(config.spatial.relative_dim, EVA01_DIM),
            nn.LayerNorm(EVA01_DIM))
        self.layer_fuse = nn.Sequential(nn.LayerNorm(EVA01_DIM), nn.Linear(EVA01_DIM, EVA01_DIM))
        self.layer_fuse_mid = nn.Sequential(nn.LayerNorm(EVA01_DIM), nn.Linear(EVA01_DIM, EVA01_DIM))
        self.layer_scale = nn.Parameter(torch.tensor(0.3))
        nn.init.eye_(self.layer_fuse[1].weight)
        nn.init.zeros_(self.layer_fuse[1].bias)
        nn.init.zeros_(self.layer_fuse_mid[1].weight)
        nn.init.zeros_(self.layer_fuse_mid[1].bias)
        self.vis_norm = nn.LayerNorm(QFORMER_HIDDEN)

        # ── 解码器 + 损失 ──
        self.decoder = MistralDecoder(llm_path=config.paths.llm_path)
        self.sigmoid_contrast_loss = MultiPositiveSigmoidContrastLoss()
        self.roi_contrast_loss = ROIAttributeContrastLoss()
        self.llip_mixture = LlipMixtureTokens(embed_dim=EVA01_DIM, num_mixture=4)

        # ── PKL 检索器 ──
        _name_pkl = getattr(
            config.paths, 'name_pkl',
            '/root/autodl-tmp/wh5319/silip_clip_caption/image_captioning_final/ext_memory_lvis.pkl')
        _attr_pkl = getattr(
            config.paths, 'attr_pkl',
            '/root/autodl-tmp/wh5319/silip_clip_caption/image_captioning_final/attr_memory/attr_memory.pkl')
        _attr_map = getattr(
            config.paths, 'attr_map',
            '/root/autodl-tmp/wh5319/silip_clip_caption/image_captioning_final/attr_memory/name2attr_topk.json')

        self._pkl_retriever = None
        try:
            self._pkl_retriever = PKLMemoryRetriever(
                name_pkl_path=_name_pkl, attr_pkl_path=_attr_pkl,
                attr_map_path=_attr_map, top_k_name=8, max_attrs=4)
            if not self._pkl_retriever._loaded:
                print(f"  [v6.6.1] ⚠️  PKL检索器未就绪")
                self._pkl_retriever = None
        except Exception as e:
            print(f"  [v6.6.1] ⚠️  PKL检索器初始化失败: {e}")
            self._pkl_retriever = None

        # ── online 模式 / 视觉前端 ──
        self._online_mode = getattr(config.training, 'mode', 'offline') == 'online'
        self.vision_frontend = OnlineVisionFrontend(config) if self._online_mode else None
        self.retrieval_qformer.set_qformer(self.evcap_retrieval_qformer)

        print(f"\n  ═══ V6.6.1 架构要点 (修 v6.6 ROI feature 来源 bug) ═══")
        print(f"  [LAVIS-QF]       Q-Former: 共享 evcap_qformer (name+attr 同一空间)")
        print(f"  [LAVIS-VIS]      视觉编码器: LAVIS eva_vit_g.pth")
        print(f"  [PER-Q]          Name 检索: per-query-token 独立检索 (不变)")
        print(f"  [V6.6.1-ATTR]    ★ Attr 检索: ROI feature + 受限检索 (修 bug 后)")
        print(f"                     ROI feature: roi_q_mean (PerROIQFormer 输出 mean)")
        print(f"                     ★★★ V6.6.1 BUG FIX:")
        print(f"                       v6.6 用 retrieval_vecs (实际是全图 32q, 不对)")
        print(f"                       v6.6.1 改用 roi_q_mean (真正的 per-ROI 768d)")
        print(f"                     threshold 0.50 → 0.30 (q-former 不同步, sim 偏低)")
        print(f"  [V6.5-CAT]       name+attr 拼成 init_q (16): 前 8 name + 后 8 attr")
        print(f"  [V6.5-PROMPT]    无文字 marker: <BOS><SYS><vis_32><name_8><attr_8><INSTRUCT>")

    # ── 公共方法 ──────────────────────────────────────────────────
    def load_blip2_pretrained(self, checkpoint_path, freeze=True):
        print(f"  加载BLIP2预训练权重: {checkpoint_path}  freeze={freeze}")
        self.aux_qformer.load_pretrained(checkpoint_path)
        self.final_qformer.load_pretrained(checkpoint_path)
        self.evcap_qformer.load_pretrained(checkpoint_path)
        self.evcap_retrieval_qformer.load_pretrained(checkpoint_path)
        for p in self.evcap_retrieval_qformer.parameters():
            p.requires_grad = False
        self.retrieval_qformer.load_blip2_weights(checkpoint_path)
        if freeze:
            self.aux_qformer.freeze_query_and_kv_only()
            self.final_qformer.freeze_query_and_kv_only()

    def enable_lora(self):
        self.decoder.enable_lora()

    def set_tokenizer(self, tokenizer):
        self.decoder.set_tokenizer(tokenizer)
        embed_fn = self.decoder._get_embed_fn()
        llm_hidden = self.decoder.llm.config.hidden_size

        sp_emb = SpatialTokenEmbedder(
            embed_fn=embed_fn, tokenizer=tokenizer,
            llm_hidden=llm_hidden, out_dim=EVA01_DIM)
        self.spatial_tag_anchor.set_embedder(sp_emb)
        self.spatial_text_embedder = sp_emb

        ent_emb = EntityTextEmbedder(
            embed_fn=embed_fn, tokenizer=tokenizer,
            llm_hidden=llm_hidden, out_dim=EVA01_DIM)
        self.spatial_tag_anchor.set_entity_embedder(ent_emb)
        self.entity_text_embedder = ent_emb

        self.object_tag_embedder = ObjectTagEmbedder(
            embed_fn=embed_fn, tokenizer=tokenizer,
            llm_hidden=llm_hidden, out_dim=EVA01_DIM)

        self.name_attr_dedup.set_embed_fn(embed_fn, tokenizer, llm_hidden=llm_hidden)

        device = next(self.decoder.llm.parameters()).device
        for m in [sp_emb, ent_emb, self.object_tag_embedder]:
            m.to(device)
            for p in m.parameters():
                p.requires_grad = False

    def load_vision_encoders(self):
        if self.vision_frontend is not None:
            device = next(self.decoder.llm.parameters()).device
            self.vision_frontend.load_encoders(device)

    def extract_online(self, images, lvis_boxes=None):
        assert self.vision_frontend is not None
        device = next(self.decoder.llm.parameters()).device
        return self.vision_frontend(images, device, external_boxes=lvis_boxes)

    def _fuse_ms_patch(self, eva_patches, eva_mid_patches):
        if eva_patches is None:
            return None
        fused = self.layer_fuse(eva_patches.float())
        if eva_mid_patches is not None:
            scale = self.layer_scale.clamp(0.0, 1.0)
            fused = fused + scale * self.layer_fuse_mid(eva_mid_patches.float())
        return fused.to(eva_patches.dtype)

    @staticmethod
    def _category_ids_to_obj_names(category_ids_b):
        """[V6.6] category_ids → obj_name list"""
        names = []
        if category_ids_b is None:
            return names
        for cid in category_ids_b:
            try:
                cid_int = int(cid)
            except Exception:
                names.append('')
                continue
            nm = COCO_ID_TO_NAME.get(cid_int, '')
            if nm and nm not in INVALID_TAG_NAMES:
                names.append(nm)
            else:
                names.append('')
        return names

    # ══════════════════════════════════════════════════════════
    # ★★★ [V6.6.1 修复版] _collect_name_attr_tokens_evcap ★★★
    # ══════════════════════════════════════════════════════════
    # v6.6 bug: 用 retrieval_vecs 当 ROI feature, 但它实际是 [B, 32, 768]
    #           全图 query, 不是 per-ROI, 跟 obj_name 错位 → sim 低 → fallback
    # v6.6.1 fix: 优先用 roi_q_mean (PerROIQFormer mean, 真正的 per-ROI)
    #             threshold 0.50 → 0.30 (q-former 不同源, sim 偏低)
    # ══════════════════════════════════════════════════════════
    def _collect_name_attr_tokens_evcap(self, global_32q, B, device, eva_patches_fused,
                                         top_k_names=8, top_k_attrs=8, dtype=None,
                                         roi_q_mean=None,
                                         retrieval_vecs=None,
                                         category_ids=None):
        K_name = top_k_names
        K_attr = top_k_attrs
        K_total = K_name + K_attr           # 16
        D = QFORMER_HIDDEN
        dtype = dtype or global_32q.dtype

        name_tokens = torch.zeros(B, K_name, D, device=device, dtype=dtype)
        attr_tokens = torch.zeros(B, K_attr, D, device=device, dtype=dtype)
        name_mask = torch.zeros(B, K_name, dtype=torch.bool, device=device)
        attr_mask = torch.zeros(B, K_attr, dtype=torch.bool, device=device)
        names_batch = [[] for _ in range(B)]
        attr_pieces_batch = [[] for _ in range(B)]
        roi_attr_diag_batch = [[] for _ in range(B)]

        self._last_vote_topk = [[] for _ in range(B)]
        self._last_vote_denoised = [[] for _ in range(B)]
        self._last_per_query_top1_batch = [[] for _ in range(B)]
        self._last_retrieval_stats_batch = [{} for _ in range(B)]

        if (self._pkl_retriever is None or not self._pkl_retriever._loaded
                or self._pkl_retriever.name_keys is None):
            self._last_roi_attr_diag = roi_attr_diag_batch
            return name_tokens, attr_tokens, name_mask, attr_mask, names_batch, attr_pieces_batch

        name_keys_t = self._pkl_retriever.name_keys.to(device).float()

        per_image_init_q = []
        per_image_n_name = []
        per_image_n_attr = []

        for b in range(B):
            # ────────────────────────────────────────────────────
            # (1) per-query name 检索 (v6.5 不变)
            # ────────────────────────────────────────────────────
            names_b, sims_b = self._pkl_retriever.retrieve_names_per_query(
                global_32q[b], top_k_per_query=5, sub_top_k=K_name,
                dedupe=True, device=device)

            self._last_per_query_top1_batch[b] = list(
                getattr(self._pkl_retriever, '_last_per_query_top1', []))
            self._last_retrieval_stats_batch[b] = {
                'thresh_pass':   getattr(self._pkl_retriever, '_last_num_thresh_pass', 0),
                'unique_names':  getattr(self._pkl_retriever, '_last_num_unique_names', 0),
                'sim_threshold': self._pkl_retriever.name_sim_threshold,
            }
            self._last_vote_topk[b] = [(n, f"{s:.2f}") for n, s in zip(names_b, sims_b)]
            self._last_vote_denoised[b] = names_b[:K_name]
            names_batch[b] = names_b[:K_name]

            # ────────────────────────────────────────────────────
            # (2) name → 768d 向量 (v6.5 不变)
            # ────────────────────────────────────────────────────
            name_embeds_b = []
            valid_names_b = []
            for name in names_b[:K_name]:
                if name in self._pkl_retriever._name2idx:
                    idx = self._pkl_retriever._name2idx[name]
                    name_embeds_b.append(name_keys_t[idx])
                    valid_names_b.append(name)

            # ════════════════════════════════════════════════════
            # (3) ★★★ V6.6.1 BUG FIX: ROI 受限 attr 检索 ★★★
            # ════════════════════════════════════════════════════
            attr_embeds_b = []
            attr_pieces_b = []
            roi_attr_diag_b = []

            # [V6.6.1 FIX] 优先用 roi_q_mean (PerROIQFormer 输出 mean, 真正 per-ROI 768d)
            #              v6.6 用的 retrieval_vecs 实际是 [B, 32, 768] 全图 query, 不是 per-ROI
            roi_query_source = None
            roi_query_label  = "none"
            if roi_q_mean is not None:
                if roi_q_mean.dim() == 3 and b < roi_q_mean.shape[0]:
                    roi_query_source = roi_q_mean[b].float()      # [N_roi, 768]
                    roi_query_label  = "roi_q_mean[B,N,768]"
                elif roi_q_mean.dim() == 2 and roi_q_mean.shape[-1] == D:
                    # 单 batch 的边界情况
                    roi_query_source = roi_q_mean.float()
                    roi_query_label  = "roi_q_mean[N,768]"

            # [V6.6.1] 如果 roi_q_mean 不可用 (non-ROI-Align fallback 路径),
            # 不再退到 retrieval_vecs (因为它形状不对). 直接走 v6.5 老路径.

            roi_path_succeeded = False
            if (self._pkl_retriever.attr_keys is not None
                    and roi_query_source is not None
                    and category_ids is not None
                    and b < len(category_ids)):
                try:
                    roi_obj_names_b = self._category_ids_to_obj_names(category_ids[b])
                    n_roi_use = min(roi_query_source.shape[0],
                                    len(roi_obj_names_b),
                                    K_attr)
                    if n_roi_use > 0:
                        roi_results = self._pkl_retriever.retrieve_attrs_per_roi_constrained(
                            roi_features=roi_query_source[:n_roi_use],
                            roi_obj_names=roi_obj_names_b[:n_roi_use],
                            threshold=0.30,            # [V6.6.1] 0.50 → 0.30
                            max_attrs_per_roi=1,
                            fallback_global=False,
                            collect_diag=False,
                        )
                        for r in roi_results:
                            if len(attr_embeds_b) >= K_attr:
                                break
                            attr_embeds_b.append(r['attr_key'].to(device).float())
                            attr_pieces_b.append(f"{r['obj_name']}:{r['attr_value']}")
                            roi_attr_diag_b.append({
                                'obj':   r['obj_name'],
                                'value': r['attr_value'],
                                'dim':   r['dim'],
                                'sim':   r['sim'],
                                'src':   roi_query_label,    # [V6.6.1 NEW]
                            })
                        roi_path_succeeded = (len(attr_embeds_b) > 0)
                except Exception as e:
                    print(f"  [V6.6.1] ROI 受限检索失败 (src={roi_query_label}), "
                          f"fallback v6.5 global: {e}")

            # ────────────────────────────────────────────────────
            # (3 fallback) 只在 ROI 路径不可用 (推理无 ROI / non-ROI-Align)
            # 或 ROI 全部不命中 (即使 0.30 阈值也没过) 时才走 v6.5 老路径
            # ────────────────────────────────────────────────────
            if not roi_path_succeeded:
                if self._pkl_retriever.attr_keys is not None and valid_names_b:
                    for name in valid_names_b[:min(4, K_name)]:
                        if len(attr_embeds_b) >= K_attr:
                            break
                        attrs_for_name = self._pkl_retriever.retrieve_attrs_per_query(
                            global_32q[b], name,
                            top_k_per_query=5, max_attrs=4, device=device)
                        for attr_info in attrs_for_name:
                            if len(attr_embeds_b) >= K_attr:
                                break
                            cand_idxs = self._pkl_retriever.name2attr_idx.get(
                                self._pkl_retriever._resolve_obj_alias(name), [])
                            for gi in cand_idxs:
                                if (gi < len(self._pkl_retriever.attr_dims)
                                        and gi < len(self._pkl_retriever.attr_values)
                                        and self._pkl_retriever.attr_dims[gi] == attr_info['dim']
                                        and self._pkl_retriever.attr_values[gi].strip() == attr_info['value']):
                                    attr_embeds_b.append(
                                        self._pkl_retriever.attr_keys[gi].to(device).float())
                                    attr_pieces_b.append(f"{name}:{attr_info['value']}")
                                    roi_attr_diag_b.append({
                                        'obj':      name,
                                        'value':    attr_info['value'],
                                        'dim':      attr_info['dim'],
                                        'sim':      attr_info.get('sim', -1.0),
                                        'fallback': True,
                                        'src':      'v6.5_global',
                                    })
                                    break

            attr_pieces_batch[b] = attr_pieces_b
            roi_attr_diag_batch[b] = roi_attr_diag_b

            # ────────────────────────────────────────────────────
            # (4) 构造定长 init_q[K_total]: 前 8 name + 后 8 attr (v6.5 不变)
            # ────────────────────────────────────────────────────
            init_q_b = torch.zeros(K_total, D, device=device, dtype=eva_patches_fused.dtype)
            n_name = min(len(name_embeds_b), K_name)
            n_attr = min(len(attr_embeds_b), K_attr)
            for i in range(n_name):
                init_q_b[i] = name_embeds_b[i].to(eva_patches_fused.dtype)
            for j in range(n_attr):
                init_q_b[K_name + j] = attr_embeds_b[j].to(eva_patches_fused.dtype)

            per_image_init_q.append(init_q_b)
            per_image_n_name.append(n_name)
            per_image_n_attr.append(n_attr)

        # ────────────────────────────────────────────────────────
        # (5) batch forward: 一次 evcap_qformer 处理 [B, 16, 768] (v6.5 不变)
        # ────────────────────────────────────────────────────────
        if per_image_init_q:
            init_q_batch = torch.stack(per_image_init_q, dim=0)
            try:
                with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=False):
                    refined_batch = self.evcap_qformer.forward_with_queries(
                        init_q_batch, eva_patches_fused.float())

                if refined_batch.shape[1] >= K_total:
                    name_tokens = refined_batch[:, :K_name].to(dtype).contiguous()
                    attr_tokens = refined_batch[:, K_name:K_total].to(dtype).contiguous()
                else:
                    n_out = refined_batch.shape[1]
                    take_n = min(K_name, n_out)
                    name_tokens[:, :take_n] = refined_batch[:, :take_n].to(dtype)
                    if n_out > K_name:
                        take_a = min(K_attr, n_out - K_name)
                        attr_tokens[:, :take_a] = refined_batch[:, K_name:K_name + take_a].to(dtype)

                for b in range(B):
                    if per_image_n_name[b] > 0:
                        name_mask[b, :per_image_n_name[b]] = True
                    if per_image_n_attr[b] > 0:
                        attr_mask[b, :per_image_n_attr[b]] = True

            except Exception as e:
                print(f"  [V6.6.1] _collect forward_with_queries fail: {e}, fallback raw embeds")
                for b in range(B):
                    nn_b = per_image_n_name[b]
                    na_b = per_image_n_attr[b]
                    if nn_b > 0:
                        name_tokens[b, :nn_b] = per_image_init_q[b][:nn_b].to(dtype)
                        name_mask[b, :nn_b] = True
                    if na_b > 0:
                        attr_tokens[b, :na_b] = per_image_init_q[b][K_name:K_name + na_b].to(dtype)
                        attr_mask[b, :na_b] = True

        self._last_roi_attr_diag = roi_attr_diag_batch

        return name_tokens, attr_tokens, name_mask, attr_mask, names_batch, attr_pieces_batch

    def _collect_name_tokens_evcap(self, global_32q, B, device, eva_patches_fused,
                                   top_k_names=8, max_attrs=4, dtype=None, roi_q_mean=None):
        """[v6.3 兼容] 只返回 name_tokens 和 names_batch (旧接口)"""
        name_tokens, _, _, _, names_batch, _ = self._collect_name_attr_tokens_evcap(
            global_32q, B, device, eva_patches_fused,
            top_k_names=top_k_names, top_k_attrs=8, dtype=dtype,
            roi_q_mean=roi_q_mean)
        return name_tokens, names_batch

    def format_evcap_retrieval_diag(self, b: int = 0, show_queries: int = 8) -> str:
        lines = []
        if not hasattr(self, '_last_per_query_top1_batch'):
            return "[EVCAP-RETR] 无诊断信息"
        if b >= len(self._last_per_query_top1_batch):
            return f"[EVCAP-RETR] batch 索引 {b} 越界"

        per_q = self._last_per_query_top1_batch[b]
        stats = self._last_retrieval_stats_batch[b] if b < len(self._last_retrieval_stats_batch) else {}
        topk = self._last_vote_topk[b] if self._last_vote_topk and b < len(self._last_vote_topk) else []
        denoised = self._last_vote_denoised[b] if self._last_vote_denoised and b < len(self._last_vote_denoised) else []
        attr_pieces = self._last_attr_pieces[b] if self._last_attr_pieces and b < len(self._last_attr_pieces) else []
        roi_attr_diag = (self._last_roi_attr_diag[b]
                         if self._last_roi_attr_diag and b < len(self._last_roi_attr_diag) else [])

        thresh = stats.get('sim_threshold', 0.15)
        n_pass = stats.get('thresh_pass', 0)
        n_uniq = stats.get('unique_names', 0)

        lines.append(
            f"  ┌─ [EVCAP-RETR] 样本{b}  per-query 检索诊断 "
            f"(阈值={thresh:.2f}, 通过={n_pass}, 独立={n_uniq})")
        if per_q:
            lines.append(f"  │  Q-Former 32个 query token 的 top-1 名称 (前{show_queries}个):")
            for qi, name, sim in per_q[:show_queries]:
                tag = "✓" if sim >= thresh else "✗"
                lines.append(f"  │    [q{qi:02d}] {name:<28s} sim={sim:.3f} {tag}")
            if len(per_q) > show_queries:
                lines.append(f"  │    ... 还有 {len(per_q) - show_queries} 个 query token")
        if topk:
            topk_str = ", ".join(f"{n}({s})" for n, s in topk[:8])
            lines.append(f"  │  合并排序去重 top-K: [{topk_str}]")
        if denoised:
            lines.append(f"  │  最终注入 LLM (name): [{', '.join(denoised[:8])}]")
        if attr_pieces:
            lines.append(f"  │  最终注入 LLM (attr): [{', '.join(attr_pieces[:8])}]")
        # [V6.6.1] ROI 受限检索 attr 详情 (含 src 标记)
        if roi_attr_diag:
            n_roi_path = sum(1 for d in roi_attr_diag if not d.get('fallback'))
            n_fallback = sum(1 for d in roi_attr_diag if d.get('fallback'))
            lines.append(f"  │  [V6.6.1 attr] {len(roi_attr_diag)} 命中 "
                         f"(ROI路径={n_roi_path}, fallback={n_fallback}):")
            for d in roi_attr_diag[:8]:
                tag = " [FALLBACK]" if d.get('fallback') else f" [{d.get('src','?')}]"
                lines.append(f"  │    {d['obj']:<14s} {d['dim']:<14s} = "
                             f"{d['value']:<16s} sim={d['sim']:.3f}{tag}")
        lines.append(f"  └─")
        return "\n".join(lines)

    def _collect_retrieval_names_and_attrs(self, B, category_ids=None, tag_sequences=None,
                                            is_core=None, eva_global_768=None, eva_rois=None,
                                            tag_confidence=None, eva_global_per_roi=None,
                                            eva_global_per_roi_alt=None, retrieval_vecs=None,
                                            fused_query=None):
        all_names, all_attrs = [], []
        for b in range(B):
            raw_names, name_attr_map, seen = [], {}, set()
            query_np = None
            if retrieval_vecs is not None and b < retrieval_vecs.shape[0]:
                q = F.normalize(retrieval_vecs[b].float().mean(0), p=2, dim=-1)
                query_np = q.float().detach().cpu().numpy()
            elif eva_global_768 is not None and b < eva_global_768.shape[0]:
                q = F.normalize(eva_global_768[b].float(), p=2, dim=-1)
                query_np = q.float().detach().cpu().numpy()
            conf_b = None
            if tag_confidence is not None:
                if hasattr(tag_confidence, 'shape') and b < tag_confidence.shape[0]:
                    conf_b = tag_confidence[b]
                elif isinstance(tag_confidence, (list, tuple)) and b < len(tag_confidence):
                    conf_b = tag_confidence[b]
            if category_ids and b < len(category_ids):
                gt_counts, gt_best_roi = {}, {}
                for i, cid in enumerate(category_ids[b]):
                    coco_name = COCO_ID_TO_NAME.get(int(cid), "")
                    if not coco_name or coco_name in INVALID_TAG_NAMES:
                        continue
                    roi_conf = 1.0
                    if conf_b is not None:
                        try:
                            roi_conf = float(conf_b[i]) if i < len(conf_b) else 0.0
                        except Exception:
                            roi_conf = 0.0
                    if roi_conf < 0.5:
                        continue
                    gt_counts[coco_name] = gt_counts.get(coco_name, 0) + 1
                    if coco_name not in gt_best_roi:
                        gt_best_roi[coco_name] = i
                for gt_name in sorted(gt_counts, key=lambda n: -gt_counts[n]):
                    if len(raw_names) >= 8:
                        break
                    count = gt_counts[gt_name]
                    roi_idx = gt_best_roi[gt_name]
                    roi_query_np = query_np
                    if (retrieval_vecs is not None and b < retrieval_vecs.shape[0]
                            and roi_idx < retrieval_vecs.shape[1]):
                        rv_roi = retrieval_vecs[b, roi_idx].float()
                        roi_query_np = F.normalize(rv_roi, p=2, dim=-1).float().detach().cpu().numpy()
                    attrs = []
                    if self._pkl_retriever is not None and roi_query_np is not None:
                        try:
                            attrs = self._pkl_retriever.retrieve_attrs(roi_query_np, gt_name)
                        except Exception:
                            pass
                    display = (f"{num_prefix(count)}{pluralize(gt_name)}"
                               if count >= 2 else gt_name)
                    raw_names.append(display.strip())
                    seen.add(gt_name)
                    name_attr_map[display.strip()] = attrs
            all_names.append(raw_names)
            all_attrs.append(name_attr_map)
        return all_names, all_attrs

    def _build_attr_feats_simple(self, eva_rois, category_ids, tag_sequences, retrieval_vecs=None):
        if not hasattr(self, 'object_tag_embedder') or eva_rois is None:
            return None
        B, N, D = eva_rois.shape
        device = eva_rois.device
        _pkl_ok = (self._pkl_retriever is not None
                   and self._pkl_retriever._loaded
                   and self._pkl_retriever.name_keys is not None)
        attr_texts = []
        for b in range(B):
            for n in range(N):
                pkl_name = ""
                pkl_sim = 0.0
                if (_pkl_ok and retrieval_vecs is not None
                        and b < retrieval_vecs.shape[0] and n < retrieval_vecs.shape[1]):
                    try:
                        rv = retrieval_vecs[b, n].float()
                        rv_np = F.normalize(rv, p=2, dim=-1).float().detach().cpu().numpy()
                        result = self._pkl_retriever.retrieve_name_topk(rv_np, topk=1)
                        if result:
                            pkl_name, pkl_sim = result[0]
                    except Exception:
                        pass
                coco_name = ""
                if category_ids and b < len(category_ids) and n < len(category_ids[b]):
                    cid = int(category_ids[b][n])
                    coco_name = COCO_ID_TO_NAME.get(cid, "")
                    if coco_name in INVALID_TAG_NAMES:
                        coco_name = ""
                tag_name = ""
                if tag_sequences and b < len(tag_sequences) and n < len(tag_sequences[b]):
                    tag = tag_sequences[b][n].strip()
                    if tag and tag.lower() not in INVALID_TAG_NAMES_EXTENDED:
                        tag_name = tag
                if pkl_name and pkl_sim >= 0.15:
                    attr_str = (coco_name if (coco_name and len(coco_name.split()) >= len(pkl_name.split()))
                                else pkl_name)
                elif coco_name:
                    attr_str = coco_name
                else:
                    attr_str = tag_name
                attr_texts.append(attr_str)
        try:
            embeds_flat = []
            for i in range(0, len(attr_texts), 64):
                batch_embed = self.object_tag_embedder.encode_batch(attr_texts[i:i + 64], device)
                embeds_flat.append(batch_embed.squeeze(1))
            return torch.cat(embeds_flat, dim=0).reshape(B, N, D).to(eva_rois.dtype)
        except Exception:
            return None

    # ── 主视觉构建 ────────────────────────────────────────────────
    def build_visual_tokens(self, eva_global, eva_global_768, eva_rois, abs_pos=None, rel_pos=None,
                            attr_feats=None, tag_sequences=None, is_core=None, importance_scores=None,
                            roi_mask=None, use_qformer=True, use_spatial_anchor=True, category_ids=None,
                            return_intermediates=False,
                            eva_patches_fused=None, eva_mid_patches=None,
                            eva_patches_raw=None, boxes_batch=None, name_tokens_768=None,
                            attr_tokens_768=None, name_mask=None,
                            pre_global_32q=None,
                            retrieval_vecs=None):
        B = eva_rois.shape[0]
        device = eva_global.device if eva_global is not None else eva_global_768.device
        global_feat = self.vis_enhancer(eva_global if eva_global is not None else eva_rois.mean(1))
        eva_global_768_for_loss = (
            self.vis_enhancer.forward_for_loss(eva_global if eva_global is not None else eva_rois.mean(1))
            if return_intermediates else None)
        use_roi_align_path = (eva_patches_raw is not None and boxes_batch is not None)

        if use_roi_align_path:
            roi_49 = self.roi_align_pooler(eva_patches_raw, boxes_batch)
            N_roi = roi_49.shape[1]
            if attr_feats is not None:
                roi_49 = self.roi_name_injector(roi_49, attr_feats[:, :N_roi])
            roi_q = self.per_roi_qformer(roi_49)
            is_core_agg = is_core[:, :N_roi] if is_core is not None else None
            imp_agg = importance_scores[:, :N_roi] if importance_scores is not None else None
            vis_32q = self.vis_norm(self.roi_token_aggregator(roi_q, imp_agg, is_core_agg))
            roi_q_mean_for_retrieval = roi_q.float().mean(dim=2)
            roi_processed_for_loss = F.normalize(vis_32q, p=2, dim=-1)
            final_32q_768 = vis_32q
            llip_g = F.normalize(global_feat.float(), p=2, dim=-1)
            fusion_kv_1408 = eva_rois
        else:
            if boxes_batch is not None and is_core is not None and importance_scores is not None:
                exp_rois, exp_is_core, exp_scores = self.explicit_roi_dedup.deduplicate(
                    eva_rois, boxes_batch, is_core, importance_scores)
            else:
                exp_rois = eva_rois
                exp_is_core = (is_core if is_core is not None
                               else torch.zeros(B, eva_rois.shape[1], dtype=torch.bool, device=device))
                exp_scores = (importance_scores if importance_scores is not None
                              else torch.ones(B, eva_rois.shape[1], device=device))
            roi_tok = self.vis_enhancer(exp_rois)
            roi_tok = self.global_ctx_fusion(roi_tok, global_feat)
            if (use_spatial_anchor and attr_feats is not None
                    and abs_pos is not None and rel_pos is not None):
                N_exp = exp_rois.shape[1]
                roi_tok = self.spatial_tag_anchor(
                    roi_tok, attr_feats[:, :N_exp], abs_pos[:, :N_exp],
                    rel_pos[:, :N_exp, :N_exp],
                    tag_sequences=tag_sequences, is_core=exp_is_core)
            roi_tok, _ = self.core_attention(roi_tok, exp_is_core, exp_scores)
            roi_processed_for_loss = F.normalize(roi_tok, p=2, dim=-1)
            if exp_is_core is not None:
                boost = self.roi_core_boost.clamp(min=1.0, max=3.0)
                roi_tok = roi_tok * torch.where(
                    exp_is_core.unsqueeze(-1),
                    torch.full_like(roi_tok, float(boost)),
                    torch.ones_like(roi_tok))
            N_exp = exp_rois.shape[1]
            sp_parts = []
            if abs_pos is not None:
                sp_parts.append(self.abs_pos_proj(abs_pos[:, :N_exp].float()))
            if rel_pos is not None:
                N_r = min(rel_pos.shape[1], N_exp)
                if N_r > 1:
                    diag = ~torch.eye(N_r, dtype=torch.bool, device=device)
                    rel_avg = (rel_pos[:, :N_r, :N_r].float()
                               * diag.unsqueeze(0).unsqueeze(-1)).sum(2) / max(N_r - 1, 1)
                else:
                    rel_avg = rel_pos[:, :N_r, :N_r].float().mean(2)
                sp_parts.append(self.rel_pos_proj(rel_avg))
            spatial_tokens = (sp_parts[0] if len(sp_parts) == 1
                              else (sp_parts[0] + sp_parts[1]) if len(sp_parts) > 1 else None)
            fusion_kv_1408 = self.global_roi_fusion(
                global_feat=global_feat, fused_rois=roi_tok,
                is_core=exp_is_core, importance_scores=exp_scores,
                spatial_tokens=spatial_tokens)
            final_32q_768 = self.final_qformer(fusion_kv_1408)
            query_in_768 = self.final_qformer.query_tokens.expand(B, -1, -1).float()
            final_32q_768 = self.aoa_gate(final_32q_768, query_in_768)
            vis_32q = self.vis_norm(final_32q_768)
            llip_g = F.normalize(global_feat.float(), p=2, dim=-1)
            roi_q_mean_for_retrieval = None

        # ── 检索 Q-Former 输出 (32q) ──
        global_32q = pre_global_32q if pre_global_32q is not None else None
        if global_32q is None and eva_patches_fused is not None:
            try:
                with torch.no_grad():
                    _ret_out, _ = self.retrieval_qformer(eva_patches_fused.float())
                global_32q = _ret_out
            except Exception:
                pass

        # ── name + attr 双通路精炼 ──
        name_tokens_from_retrieval = None
        attr_tokens_from_retrieval = None
        name_mask_ret = None
        attr_mask_ret = None
        self._last_implicit_names = [[] for _ in range(B)]
        self._last_attr_pieces = [[] for _ in range(B)]

        if (global_32q is not None and eva_patches_fused is not None
                and self._pkl_retriever is not None and self._pkl_retriever._loaded):
            (name_tokens_from_retrieval, attr_tokens_from_retrieval,
             name_mask_ret, attr_mask_ret,
             self._last_implicit_names, self._last_attr_pieces) = \
                self._collect_name_attr_tokens_evcap(
                    global_32q, B, device, eva_patches_fused,
                    top_k_names=8, top_k_attrs=8, dtype=vis_32q.dtype,
                    roi_q_mean=roi_q_mean_for_retrieval if use_roi_align_path else None,
                    retrieval_vecs=retrieval_vecs,
                    category_ids=category_ids)

        # ── 三流融合 (global + vis + name) ──
        if global_32q is not None and hasattr(self, 'tri_stream_fusion'):
            vis_tokens_768 = self.tri_stream_fusion(
                global_32q.to(vis_32q.dtype), vis_32q, name_tokens_from_retrieval)
        else:
            vis_tokens_768 = vis_32q

        if return_intermediates:
            return (vis_tokens_768, name_tokens_from_retrieval, attr_tokens_from_retrieval,
                    name_mask_ret, attr_mask_ret,
                    final_32q_768, eva_global_768_for_loss, roi_processed_for_loss, llip_g,
                    fusion_kv_1408.detach() if hasattr(fusion_kv_1408, 'detach') else fusion_kv_1408,
                    torch.tensor(0.0, device=device), torch.tensor(0.0, device=device))
        return (vis_tokens_768, name_tokens_from_retrieval, attr_tokens_from_retrieval,
                name_mask_ret, attr_mask_ret)

    # ── 训练 forward ──────────────────────────────────────────────
    def forward(self, eva_global, eva_global_768, eva_rois, text_tok, text_mask=None,
                abs_pos=None, rel_pos=None, attr_feats=None, tag_sequences=None,
                is_core=None, importance_scores=None, roi_mask=None, category_ids=None,
                use_qformer=True, use_spatial_anchor=True, return_intermediates=False,
                eva_patches=None, eva_mid_patches=None,
                eva_patch_single=None,
                tag_confidence=None, eva_global_per_roi=None,
                eva_global_per_roi_alt=None, boxes_batch=None, **kwargs):
        B = eva_rois.shape[0]
        device = eva_rois.device
        retrieval_compressed = retrieval_vecs = None
        if eva_patches is not None:
            try:
                retrieval_compressed, retrieval_vecs = self.retrieval_qformer(eva_patches)
            except Exception:
                pass
        # 多尺度融合 (EVA last + EVA mid → fused patches, 给检索 Q-Former 用)
        eva_patches_fused = self._fuse_ms_patch(eva_patches, eva_mid_patches)
        if attr_feats is None or attr_feats.abs().sum() == 0:
            attr_feats = self._build_attr_feats_simple(
                eva_rois, category_ids, tag_sequences, retrieval_vecs=retrieval_vecs)
        build_out = self.build_visual_tokens(
            eva_global, eva_global_768, eva_rois, abs_pos=abs_pos, rel_pos=rel_pos,
            attr_feats=attr_feats, tag_sequences=tag_sequences, is_core=is_core,
            importance_scores=importance_scores, roi_mask=roi_mask,
            use_qformer=use_qformer, use_spatial_anchor=use_spatial_anchor,
            category_ids=category_ids, return_intermediates=return_intermediates,
            eva_patches_fused=eva_patches_fused, eva_mid_patches=eva_mid_patches,
            eva_patches_raw=eva_patches, boxes_batch=boxes_batch,
            pre_global_32q=retrieval_compressed,
            retrieval_vecs=retrieval_vecs)
        if return_intermediates:
            (vis_tokens_768, object_name_features_768, attr_features_768,
             name_mask_ret, attr_mask_ret,
             final_32q, eva_global_768_for_loss, roi_proc, llip_g,
             fusion_deco, mask_loss, mat_loss) = build_out
        else:
            (vis_tokens_768, object_name_features_768, attr_features_768,
             name_mask_ret, attr_mask_ret) = build_out
            final_32q = eva_global_768_for_loss = roi_proc = llip_g = fusion_deco = None
            mask_loss = mat_loss = torch.tensor(0.0, device=device)
        logits = self.decoder(
            vis_tokens_768, text_tok, text_mask,
            name_tokens_768=object_name_features_768,
            attr_tokens_768=attr_features_768,
            name_mask=name_mask_ret,
            attr_mask=attr_mask_ret,
        )
        if return_intermediates:
            return (logits, final_32q, eva_global_768_for_loss, roi_proc, llip_g,
                    fusion_deco, mask_loss, mat_loss, torch.tensor(0.0, device=device))
        return logits, torch.tensor(0.0, device=device)

    @torch.no_grad()
    def generate(self, eva_global, eva_global_768, eva_rois, abs_pos=None, rel_pos=None,
                 attr_feats=None, tag_sequences=None, is_core=None, importance_scores=None,
                 roi_mask=None, category_ids=None, use_qformer=True, use_spatial_anchor=True,
                 eva_patches=None, eva_mid_patches=None,
                 eva_patch_single=None,
                 tag_confidence=None, eva_global_per_roi=None,
                 eva_global_per_roi_alt=None, boxes_batch=None, **kwargs):
        B = eva_rois.shape[0]
        device = eva_rois.device
        retrieval_compressed = retrieval_vecs = None
        if eva_patches is not None:
            try:
                retrieval_compressed, retrieval_vecs = self.retrieval_qformer(eva_patches)
            except Exception:
                pass
        eva_patches_fused = self._fuse_ms_patch(eva_patches, eva_mid_patches)
        if attr_feats is None or attr_feats.abs().sum() == 0:
            attr_feats = self._build_attr_feats_simple(
                eva_rois, category_ids, tag_sequences, retrieval_vecs=retrieval_vecs)
        (vis_tokens_768, name_tokens_for_llm, attr_tokens_for_llm,
         name_mask_ret, attr_mask_ret) = self.build_visual_tokens(
            eva_global, eva_global_768, eva_rois, abs_pos=abs_pos, rel_pos=rel_pos,
            attr_feats=attr_feats, tag_sequences=tag_sequences, is_core=is_core,
            importance_scores=importance_scores, roi_mask=roi_mask,
            use_qformer=use_qformer, use_spatial_anchor=use_spatial_anchor,
            category_ids=category_ids, return_intermediates=False,
            eva_patches_fused=eva_patches_fused, eva_mid_patches=eva_mid_patches,
            eva_patches_raw=eva_patches, boxes_batch=boxes_batch,
            pre_global_32q=retrieval_compressed,
            retrieval_vecs=retrieval_vecs)
        for k in ('category_ids', 'tag_sequences', 'is_core', 'importance_scores',
                  'object_names', 'use_reranking', 'clip_text_feat', 'tag_confidence',
                  'eva_rois_raw', 'eva_rois_alt', 'kp_tokens',
                  'eva_global_per_roi', 'eva_global_per_roi_alt',
                  'boxes_batch', 'implicit_rois',
                  'name_tokens', 'name_tokens_768', 'attr_tokens', 'attr_tokens_768',
                  'name_mask', 'attr_mask'):
            kwargs.pop(k, None)
        return self.decoder.generate(
            vis_tokens_768,
            name_tokens_768=name_tokens_for_llm,
            attr_tokens_768=attr_tokens_for_llm,
            name_mask=name_mask_ret,
            attr_mask=attr_mask_ret,
            **kwargs,
        )

    # ── 损失 ──────────────────────────────────────────────────────
    def compute_auxiliary_losses(self, eva_global_768_for_loss=None, roi_processed_for_loss=None,
                                  llip_global_normed=None, text_feat_normed=None, img_ids=None,
                                  w_sigmoid=0.1, w_roi=0.05, **kwargs):
        if eva_global_768_for_loss is None:
            return {'total': torch.tensor(0.0, device=next(self.parameters()).device)}
        dev = eva_global_768_for_loss.device
        aux = torch.tensor(0.0, device=dev)
        details = {}
        sig_loss = torch.tensor(0.0, device=dev)
        if text_feat_normed is not None and w_sigmoid > 0:
            try:
                raw_repr = F.normalize(eva_global_768_for_loss.float(), p=2, dim=-1)
                try:
                    _, mix_repr = self.llip_mixture.forward_for_contrast(
                        eva_global_768_for_loss.float().unsqueeze(1))
                    img_repr = F.normalize(0.7 * raw_repr + 0.3 * mix_repr, p=2, dim=-1)
                except Exception:
                    img_repr = raw_repr
                sig_loss = self.sigmoid_contrast_loss(img_repr, text_feat_normed, img_ids)
                aux = aux + w_sigmoid * sig_loss
            except Exception:
                pass
        details['SigmoidContrastLoss'] = float(sig_loss.item())
        roi_loss = torch.tensor(0.0, device=dev)
        if roi_processed_for_loss is not None and w_roi > 0:
            try:
                B_, N_, _ = roi_processed_for_loss.shape
                if B_ > 1 and N_ > 1:
                    roi_loss = self.roi_contrast_loss(roi_processed_for_loss)
                    aux = aux + w_roi * roi_loss
            except Exception:
                pass
        details['ROIAttributeContrastLoss'] = float(roi_loss.item())
        details['total'] = aux
        return details

    def decode_and_clean(self, token_ids, tokenizer):
        if hasattr(token_ids, 'tolist'):
            token_ids = token_ids.tolist()
        return clean_caption(tokenizer.decode(token_ids, skip_special_tokens=True).strip())

    # ── 参数冻结/解冻 (v6.5 完全保留) ─────────────────────────────

    def freeze_stage1_only(self):
        for p in self.parameters():
            p.requires_grad = False
        for p in self.vis_enhancer.parameters():
            p.requires_grad = True
        for p in self.decoder.vis_proj_A.parameters():
            p.requires_grad = True
        n = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"  [Stage1] 可训练: {n / 1e6:.2f}M")

    def freeze_for_soft_stage1(self):
        for p in self.parameters():
            p.requires_grad = False
        for p in self.evcap_qformer.output_proj.parameters():
            p.requires_grad = True
        for p in self.evcap_qformer.query_in_proj.parameters():
            p.requires_grad = True
        self.evcap_qformer.query_tokens.requires_grad = True
        for p in self.attr_qformer.output_proj.parameters():
            p.requires_grad = True
        for p in self.attr_qformer.query_in_proj.parameters():
            p.requires_grad = True
        self.attr_qformer.query_tokens.requires_grad = True
        for p in self.tri_stream_fusion.q2.parameters():
            p.requires_grad = True
        for p in self.tri_stream_fusion.k2.parameters():
            p.requires_grad = True
        for p in self.tri_stream_fusion.v2.parameters():
            p.requires_grad = True
        for p in self.tri_stream_fusion.out2.parameters():
            p.requires_grad = True
        for p in self.tri_stream_fusion.gate.parameters():
            p.requires_grad = True
        for p in self.tri_stream_fusion.norm3.parameters():
            p.requires_grad = True
        for p in self.decoder.vis_proj_A.parameters():
            p.requires_grad = True
        n = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"  [SoftStage1-v6.6.1] 可训练: {n / 1e6:.2f}M")

    def unfreeze_for_stage2(self):
        for p in self.parameters():
            p.requires_grad = False
        for p in self.decoder.vis_proj_A.parameters():
            p.requires_grad = True
        for p in self.vis_norm.parameters():
            p.requires_grad = True
        self.layer_scale.requires_grad = True
        for p in self.roi_token_aggregator.parameters():
            p.requires_grad = True
        if hasattr(self, 'tri_stream_fusion'):
            for p in self.tri_stream_fusion.parameters():
                p.requires_grad = True
        for p in self.evcap_qformer.output_proj.parameters():
            p.requires_grad = True
        for p in self.evcap_qformer.query_in_proj.parameters():
            p.requires_grad = True
        self.evcap_qformer.query_tokens.requires_grad = True
        for p in self.attr_qformer.output_proj.parameters():
            p.requires_grad = True
        for p in self.attr_qformer.query_in_proj.parameters():
            p.requires_grad = True
        self.attr_qformer.query_tokens.requires_grad = True
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = sum(p.numel() for p in self.parameters() if not p.requires_grad)
        print(f"  [Stage2 v6.6.1] 可训练: {trainable / 1e6:.2f}M, 冻结: {frozen / 1e6:.2f}M")

    def unfreeze_new_modules(self):
        newly = 0
        if hasattr(self, 'tri_stream_fusion'):
            for p in self.tri_stream_fusion.parameters():
                if not p.requires_grad:
                    p.requires_grad = True
                    newly += p.numel()
        if hasattr(self, 'attr_final_gating'):
            for p in self.attr_final_gating.parameters():
                if not p.requires_grad:
                    p.requires_grad = True
                    newly += p.numel()
        for p in self.evcap_qformer.parameters():
            if not p.requires_grad:
                p.requires_grad = True
                newly += p.numel()
        for p in self.attr_qformer.parameters():
            if not p.requires_grad:
                p.requires_grad = True
                newly += p.numel()
        if (hasattr(self, 'vision_frontend') and self.vision_frontend is not None
                and hasattr(self.vision_frontend, 'roi_refiner')):
            for p in self.vision_frontend.roi_refiner.parameters():
                if not p.requires_grad:
                    p.requires_grad = True
                    newly += p.numel()
        total = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"  [unfreeze_new_modules v6.6.1] +{newly / 1e6:.2f}M → 总可训练: {total / 1e6:.2f}M")
        return total

    def unfreeze_for_stage2_lite(self):
        for p in self.parameters():
            p.requires_grad = False
        for p in self.vis_enhancer.parameters():
            p.requires_grad = True
        for p in self.decoder.vis_proj_A.parameters():
            p.requires_grad = True
        for p in self.vis_norm.parameters():
            p.requires_grad = True
        for p in self.aoa_gate.parameters():
            p.requires_grad = True
        self.roi_core_boost.requires_grad = True
        self.final_qformer.query_tokens.requires_grad = True
        for p in self.final_qformer.ln_vision.parameters():
            p.requires_grad = True
        for p in self.final_qformer.output_proj.parameters():
            p.requires_grad = True
        for p in self.final_qformer.query_in_proj.parameters():
            p.requires_grad = True
        for layer in self.final_qformer.encoder.layers:
            if layer.has_cross_attn:
                for p in layer.cross_attn.self.key.parameters():
                    p.requires_grad = True
                for p in layer.cross_attn.self.value.parameters():
                    p.requires_grad = True
        self.evcap_qformer.query_tokens.requires_grad = True
        for p in self.evcap_qformer.ln_vision.parameters():
            p.requires_grad = True
        for p in self.evcap_qformer.output_proj.parameters():
            p.requires_grad = True
        for p in self.evcap_qformer.query_in_proj.parameters():
            p.requires_grad = True
        for layer in self.evcap_qformer.encoder.layers:
            if layer.has_cross_attn:
                for p in layer.cross_attn.self.key.parameters():
                    p.requires_grad = True
                for p in layer.cross_attn.self.value.parameters():
                    p.requires_grad = True
        self.attr_qformer.query_tokens.requires_grad = True
        for p in self.attr_qformer.ln_vision.parameters():
            p.requires_grad = True
        for p in self.attr_qformer.output_proj.parameters():
            p.requires_grad = True
        for p in self.attr_qformer.query_in_proj.parameters():
            p.requires_grad = True
        for layer in self.attr_qformer.encoder.layers:
            if layer.has_cross_attn:
                for p in layer.cross_attn.self.key.parameters():
                    p.requires_grad = True
                for p in layer.cross_attn.self.value.parameters():
                    p.requires_grad = True
        for p in self.roi_token_aggregator.parameters():
            p.requires_grad = True
        if hasattr(self, 'tri_stream_fusion'):
            for p in self.tri_stream_fusion.parameters():
                p.requires_grad = True
        if (hasattr(self, 'vision_frontend') and self.vision_frontend is not None
                and hasattr(self.vision_frontend, 'roi_refiner')):
            for p in self.vision_frontend.roi_refiner.parameters():
                p.requires_grad = True
        if hasattr(self, 'attr_final_gating'):
            for p in self.attr_final_gating.parameters():
                p.requires_grad = True
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = sum(p.numel() for p in self.parameters() if not p.requires_grad)
        print(f"  [Stage2-LITE v6.6.1] 可训练: {trainable / 1e6:.2f}M  冻结: {frozen / 1e6:.2f}M")
        return {"trainable_M": round(trainable / 1e6, 2),
                "frozen_M": round(frozen / 1e6, 2)}

    def unfreeze_global_roi_for_stage2_late(self):
        newly = 0
        for mod in [self.global_roi_fusion]:
            for p in mod.parameters():
                if not p.requires_grad:
                    p.requires_grad = True
                    newly += p.numel()
        if self.name_attr_dedup.name_proj is not None:
            for p in self.name_attr_dedup.name_proj.parameters():
                if not p.requires_grad:
                    p.requires_grad = True
                    newly += p.numel()
        if self.name_attr_dedup.attr_proj is not None:
            for p in self.name_attr_dedup.attr_proj.parameters():
                if not p.requires_grad:
                    p.requires_grad = True
                    newly += p.numel()
        total = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"  [Stage2-Late] +{newly / 1e6:.2f}M  总计: {total / 1e6:.2f}M")

    def get_trainable_param_groups(self, base_lr=3e-5, vis_proj_mult=3.0, attr_mult=1.5):
        proj_params = list(self.decoder.vis_proj_A.parameters())
        proj_ids = {id(p) for p in proj_params}

        attr_mods = []
        for mod_name in ['evcap_qformer', 'attr_qformer', 'attr_final_gating',
                         'roi_token_aggregator', 'tri_stream_fusion']:
            mod = getattr(self, mod_name, None)
            if mod is not None:
                attr_mods += list(mod.parameters())
        attr_params = [p for p in attr_mods if p.requires_grad]
        attr_ids = {id(p) for p in attr_params}

        rest_params = [p for p in self.parameters()
                       if p.requires_grad and id(p) not in proj_ids
                       and id(p) not in attr_ids]

        groups = [
            {"params": proj_params, "lr": base_lr * vis_proj_mult, "name": "vis_proj"},
            {"params": attr_params, "lr": base_lr * attr_mult,     "name": "attr_mods"},
            {"params": rest_params, "lr": base_lr,                 "name": "rest"},
        ]
        for g in groups:
            n = sum(p.numel() for p in g["params"])
            if n > 0:
                print(f"  param_group [{g['name']:16s}] lr={g['lr']:.2e}  {n / 1e6:.2f}M")
        return [g for g in groups if sum(p.numel() for p in g["params"]) > 0]

    # ════════════════════════════════════════════════════════════════
    # [LEGACY-CKPT-COMPAT] 旧 siglip_* 命名的 ckpt → 新名映射
    # ════════════════════════════════════════════════════════════════
    # 历史 ckpt 里的 state_dict key 含 'siglip_aligner.*' / 'siglip_qformer.*',
    # 新代码改名后这些 key 会变 missing. 这里在 load_state_dict 前
    # 自动把旧 key 改写成新 key, 让 --resume 旧 ckpt 仍能干净加载.
    _LEGACY_KEY_MAP = {
        'siglip_aligner.':  'vis_enhancer.',
        'siglip_qformer.':  'aux_qformer.',
    }

    @classmethod
    def _remap_legacy_state_dict_keys(cls, state_dict):
        """旧 ckpt key → 新 key. 返回 (remapped_dict, n_remapped)."""
        if not isinstance(state_dict, dict):
            return state_dict, 0
        out = {}
        n = 0
        for k, v in state_dict.items():
            new_k = k
            for old_prefix, new_prefix in cls._LEGACY_KEY_MAP.items():
                if k.startswith(old_prefix):
                    new_k = new_prefix + k[len(old_prefix):]
                    n += 1
                    break
            out[new_k] = v
        return out, n

    def load_state_dict(self, state_dict, strict=True, assign=False):
        """重写以自动迁移旧 siglip_* key. 完全兼容 nn.Module.load_state_dict 签名."""
        sd, n_remapped = self._remap_legacy_state_dict_keys(state_dict)
        if n_remapped > 0:
            print(f"  [LEGACY-CKPT] 自动迁移 {n_remapped} 个 siglip_* key "
                  f"→ 新名 (vis_enhancer.* / aux_qformer.*)")
        try:
            return super().load_state_dict(sd, strict=strict, assign=assign)
        except TypeError:
            # 老 torch 版本没有 assign 参数
            return super().load_state_dict(sd, strict=strict)


def create_model(config):
    return CaptionModel(config)