#!/usr/bin/env python3
"""
caption_model ── v6.3 图像描述模型 (拆分自原 model.py 单文件)

向后兼容旧 train.py:
    from caption_model import CaptionModel, _clean_caption

或者新风格导入:
    from caption_model import CaptionModel, create_model, clean_caption
"""

# ── 主类 ────────────────────────────────────────────
from .model import CaptionModel, create_model

# ── 解码器 / 视觉前端 ───────────────────────────────
from .decoder import MistralDecoder
from .vision_frontend import OnlineVisionFrontend

# ── Q-Former ────────────────────────────────────────
from .qformer import BLIP2QFormer
from .retrieval_qformer import RetrievalQFormer, LAVISQFormerForRetrieval, AttrQFormer

# ── 检索 ────────────────────────────────────────────
from .retriever import PKLMemoryRetriever, build_memory_bank_from_pkl
# [V6.8.1] LightSoftRetriever 已弃用 (forward 路径从未调用), 不在此导出
# 文件 caption_model/soft_retriever.py 仍保留, 想恢复可手动 import
# from .soft_retriever import LightSoftRetriever

# ── ROI 管线 ────────────────────────────────────────
from .roi_pipeline import (
    ROIAlignPooler,
    ROINameAttrInjector,
    PerROIQFormer,
    ROITokenAggregator,
    ExplicitROIDeduplicator,
)

# ── 融合 ────────────────────────────────────────────
from .fusion import (
    AttrFinalGating,
    ExplicitImplicitFusion,
    TriStreamFusion,
    GlobalROIBidirectionalFusion,
    ROICrossAttnRefinement,
)

# ── 辅助模块 ────────────────────────────────────────
from .auxiliary import (
    NameAttrDeduplicator,
    PKLAnchoredImplicitROI,
    IndexAnchoredImplicitROI,
    SpatialTokenEmbedder,
    EntityTextEmbedder,
    VisFeatureEnhancer,
    AoAGate,
    EVCapVisualProjector,
    GlobalContextFusion,
    CollectiveObjectMerger,
    TypeEmbedding,
    CoreObjectAttention,
    SpatialTagAnchor,
    LlipMixtureTokens,
    ObjectTagEmbedder,
    ContrastiveHead,
)

# ── 损失 ────────────────────────────────────────────
from .losses import (
    MultiPositiveSigmoidContrastLoss,
    ROIAttributeContrastLoss,
)

# ── 工具 / 常量 ─────────────────────────────────────
from .utils import (
    clean_caption,
    evcap_pre_name,
    abs_to_loc_tokens,
    sinusoidal_expand,
    rel_sinusoidal,
    build_entity_text,
    per_token_vote,
    patch_torch_load_safety,
)
from .constants import (
    NUM_BINS, QFORMER_HIDDEN, EVA01_DIM, RETRIEVAL_DIM, LLM_HIDDEN,
    NAME_SIM_THRESHOLD,
    INVALID_TAG_NAMES, INVALID_TAG_NAMES_EXTENDED,
    COCO_NAME_SET, COCO_ID_TO_NAME,
    NUM_WORDS_PROMPT, num_prefix, pluralize,
)

# ── 向后兼容老接口 (老 train.py 用 _clean_caption) ──
_clean_caption     = clean_caption
_evcap_pre_name    = evcap_pre_name
_abs_to_loc_tokens = abs_to_loc_tokens
_sinusoidal_expand = sinusoidal_expand
_rel_sinusoidal    = rel_sinusoidal
_build_entity_text = build_entity_text
_per_token_vote    = per_token_vote

_EVA01_DIM         = EVA01_DIM
_QFORMER_HIDDEN    = QFORMER_HIDDEN
_RETRIEVAL_DIM     = RETRIEVAL_DIM
_LLM_HIDDEN        = LLM_HIDDEN
_NUM_BINS          = NUM_BINS
_NAME_SIM_THRESHOLD = NAME_SIM_THRESHOLD
_INVALID_TAG_NAMES = INVALID_TAG_NAMES
_INVALID_TAG_NAMES_EXTENDED = INVALID_TAG_NAMES_EXTENDED
_COCO_NAME_SET     = COCO_NAME_SET
_COCO_ID_TO_NAME   = COCO_ID_TO_NAME
_NUM_WORDS_PROMPT  = NUM_WORDS_PROMPT
_num_prefix        = num_prefix
_pluralize         = pluralize


__version__ = "6.4"

__all__ = [
    # 主类
    "CaptionModel", "create_model",
    # 子模块大类
    "MistralDecoder", "OnlineVisionFrontend",
    "BLIP2QFormer", "RetrievalQFormer", "LAVISQFormerForRetrieval", "AttrQFormer",
    "PKLMemoryRetriever", "build_memory_bank_from_pkl",
    # "LightSoftRetriever",  # [V6.8.1] 已弃用
    # ROI / Fusion
    "ROIAlignPooler", "ROINameAttrInjector", "PerROIQFormer",
    "ROITokenAggregator", "ExplicitROIDeduplicator",
    "AttrFinalGating", "ExplicitImplicitFusion", "TriStreamFusion",
    "GlobalROIBidirectionalFusion", "ROICrossAttnRefinement",
    # Auxiliary
    "NameAttrDeduplicator", "PKLAnchoredImplicitROI", "IndexAnchoredImplicitROI",
    "SpatialTokenEmbedder", "EntityTextEmbedder", "VisFeatureEnhancer",
    "AoAGate", "EVCapVisualProjector", "GlobalContextFusion",
    "CollectiveObjectMerger", "TypeEmbedding", "CoreObjectAttention",
    "SpatialTagAnchor", "LlipMixtureTokens", "ObjectTagEmbedder",
    "ContrastiveHead",
    # Losses
    "MultiPositiveSigmoidContrastLoss", "ROIAttributeContrastLoss",
    # Utils
    "clean_caption", "evcap_pre_name", "abs_to_loc_tokens",
    "sinusoidal_expand", "rel_sinusoidal", "build_entity_text",
    "per_token_vote", "patch_torch_load_safety",
    # Constants
    "NUM_BINS", "QFORMER_HIDDEN", "EVA01_DIM", "RETRIEVAL_DIM", "LLM_HIDDEN",
    "NAME_SIM_THRESHOLD", "INVALID_TAG_NAMES", "INVALID_TAG_NAMES_EXTENDED",
    "COCO_NAME_SET", "COCO_ID_TO_NAME", "NUM_WORDS_PROMPT",
    "num_prefix", "pluralize",
    # 向后兼容老接口
    "_clean_caption", "_evcap_pre_name",
    "_EVA01_DIM", "_QFORMER_HIDDEN", "_LLM_HIDDEN", "_RETRIEVAL_DIM",
    "_COCO_ID_TO_NAME",
]