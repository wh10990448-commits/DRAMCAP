#!/usr/bin/env python3
"""
config_loader.py  v6.5 (Karpathy 118k 适配版)
================================================
适配: model.py v6.6.1 + train.py + losses.py + config v6.4
基础: 完全继承 v6.4 (A+B 修复, TYPE-EMB / Q-PAD / LLM-SEG)

v6.5 改动 (相对 v6.4): ★ 仅本文件改动, 其他文件结构不变
═════════════════════════════════════════════════════════════════
  [KARPATHY-PATH]  新加 5 个 paths 字段, 指向 Karpathy 118k 数据集:
                     karpathy_dir         /root/autodl-tmp/wh5319/all_data/coco2014_karpathy
                     karpathy_train_json  coco_karpathy_train.json  (★ 训练用)
                     karpathy_val_json    coco_karpathy_val.json
                     karpathy_test_json   coco_karpathy_test.json
                     karpathy_image_root  /root/autodl-tmp/wh5319/all_data/coco_2014
                                          (父目录, 含 train2014/ + val2014/)

  [DUAL-INSTANCES] ★ 新加 lvis_val_json 字段
                     原因: Karpathy split 把 COCO2014 train+val 混合切分,
                           coco_karpathy_train.json 里同时含 train2014/ 和 val2014/ 路径,
                           所以 box 标注必须同时读两个 instances JSON:
                       train2014/COCO_train2014_xxx.jpg → instances_train2014.json
                       val2014/COCO_val2014_xxx.jpg     → instances_val2014.json
                     train.py 里 CocoBoxLoader 需要按图片路径前缀分发到对应 instances

  [KARPATHY-FMT]   Karpathy JSON 格式 (扁平, 跟旧 captions_train2014.json 完全不同):
                     [{"caption": "...", "image": "val2014/xxx.jpg", "image_id": "coco_xxx"}, ...]
                   每条 = 一个 caption (一张图有 5 条记录)
                   ★ train.py 的 OnlineCaptionDataset 必须改解析逻辑, 否则报错

  [BACKWARD-COMPAT] 旧字段 coco_train_captions / coco_val_captions / lvis_train_json
                    全部保留不动, 现有的 ablation / verify / build_attr 路径不受影响

继承 v6.4 (不变):
  [TYPE-EMB]  evcap_qformer query_type_emb (0=name, 1=attr)
  [Q-PAD]     forward_with_queries 支持 query_padding_mask
  [LLM-SEG]   MistralDecoder Objects:/Attributes: 分段
  [SPLIT-NA]  _collect_name_tokens_evcap 返回 3-tuple

继承 v6.2:
  [LAVIS-PATH]      eva_vit_g_model 字段
  [COSINE-RETRIEVAL] PKL 纯余弦检索
  [ROI-ALIGN-49]    ROIAlignPooler + PerROIQFormer + ROITokenAggregator
  [TriStreamFusion] global_32q + vis_32q + cat(name+attr) 三流融合
  [LOSS-SIMPLIFY]   仅 NTP cross-entropy, 默认 label_smoothing=0.0
"""

import os
import yaml
from dataclasses import dataclass, field
from typing import Optional


def _is_property(obj, attr_name):
    for cls in type(obj).__mro__:
        if attr_name in cls.__dict__:
            if isinstance(cls.__dict__[attr_name], property):
                return True
    return False


def _apply_dict(obj, d: Optional[dict]):
    if not d:
        return
    for k, v in d.items():
        if _is_property(obj, k):
            continue
        try:
            setattr(obj, k, v)
        except (AttributeError, TypeError):
            print(f"[config] 警告: 无法设置 {type(obj).__name__}.{k}={v}")


# ─────────────────────────────────────────────────────────────
# 配置类
# ─────────────────────────────────────────────────────────────

@dataclass
class PreprocessConfig:
    # [v6.3 继承] 默认 ROI 数 9, core_count 9
    CORE: int   = 9
    MAX_ROIS: int = 9
    MIN_ROIS: int = 4
    BATCH: int  = 32
    WORKERS: int = 4
    MIN_AREA: int = 200
    MIN_SIDE: int = 10
    YOLOE26_CONF: float = 0.20
    YOLOE26_WORKERS: int = 2
    PREFETCH_BATCHES: int = 2
    USE_YOLO_POSE: bool = True
    RESUME_PREPROCESS: bool = False
    ROI_ONLINE: bool = True
    ROI_SEG_GRID: int = 12
    ROI_POOL_GRID: int = 12
    batch_size: int    = 32
    num_workers: int   = 4
    min_area: int      = 50
    min_side: int      = 10
    max_rois: int      = 9
    min_rois: int      = 4
    core_count: int    = 9
    num_categories: int = 270
    coord_patches_per_roi: int = 25
    yolo_pose_supplement: bool = True
    bf16_storage: bool = True


@dataclass
class PathsConfig:
    project_dir:    str = "/root/autodl-tmp/wh5319/silip_clip_caption"
    llm_path:       str = "/root/autodl-tmp/wh5319/silip_clip_caption/model/vicuna-7b-v1.5"
    clip_model:     str = "/root/autodl-tmp/wh5319/silip_clip_caption/clip_large_patch14"
    # [LEGACY-RENAMED] 历史死配置, 从未被加载. 真实视觉编码器是 eva_vit_g_model
    # 旧 yaml 里写 'siglip_model: ...' 会被 get_config 自动迁移到这里
    _unused_siglip_model_path: str = "/root/autodl-tmp/wh5319/silip_clip_caption/model/siglip2-base-patch16-512"
    yoloe26_ckpt:   str = "/root/autodl-tmp/wh5319/silip_clip_caption/model/yoloe26_m_seg/model.pt"
    blip2_qformer_path: str = "/root/autodl-tmp/wh5319/silip_clip_caption/model/blip2_pretrained_flant5xxl.pth"

    # ─── COCO 2014 raw 数据 (旧字段, 不变) ─────────────────────────
    coco_dir:       str = "/root/autodl-tmp/wh5319/all_data/coco_2014"
    coco_train_images:   str = "/root/autodl-tmp/wh5319/all_data/coco_2014/train2014"
    coco_train_captions: str = "/root/autodl-tmp/wh5319/all_data/coco_2014/annotations/captions_train2014.json"
    coco_val_images:     str = "/root/autodl-tmp/wh5319/all_data/coco_2014/val2014"
    coco_val_captions:   str = "/root/autodl-tmp/wh5319/all_data/coco_2014/annotations/captions_val2014.json"

    # ─── ★ [V6.5 新加] Karpathy split (118k) 数据集 ───────────────
    # 格式: [{"caption": "...", "image": "val2014/xxx.jpg", "image_id": "coco_xxx"}, ...]
    # 每条 = 一个 caption (扁平, 一张图 5 条)
    # image 字段含 train2014/ 或 val2014/ 前缀, image_root 必须指父目录
    karpathy_dir:        str = "/root/autodl-tmp/wh5319/all_data/coco2014_karpathy"
    karpathy_train_json: str = "/root/autodl-tmp/wh5319/all_data/coco2014_karpathy/coco_karpathy_train.json"
    karpathy_val_json:   str = "/root/autodl-tmp/wh5319/all_data/coco2014_karpathy/coco_karpathy_val.json"
    karpathy_test_json:  str = "/root/autodl-tmp/wh5319/all_data/coco2014_karpathy/coco_karpathy_test.json"
    karpathy_image_root: str = "/root/autodl-tmp/wh5319/all_data/coco_2014"  # 含 train2014/ + val2014/

    # 训练时数据集 / 格式开关 (train.py 用这两个决定走哪条路径)
    # dataset_format = "karpathy" → train.py 用 karpathy_train_json + karpathy_image_root
    # dataset_format = "coco"     → train.py 用 coco_train_captions  + coco_train_images (旧路径)
    dataset_format: str = "karpathy"

    # ─── Instance 标注 (LVIS / COCO instances 双源 ★) ─────────────
    # ★ Karpathy split 同时用 train2014/ 和 val2014/ 图,
    #   所以 box 标注必须同时读两个 instances JSON.
    # train.py 里 CocoBoxLoader 需要按图片路径前缀分发到对应 instances:
    #   image_rel.startswith('train2014/') → 查 lvis_train_json
    #   image_rel.startswith('val2014/')   → 查 lvis_val_json
    lvis_train_json: str = "/root/autodl-tmp/wh5319/all_data/coco_2014/annotations/instances_train2014.json"
    lvis_val_json:   str = "/root/autodl-tmp/wh5319/all_data/coco_2014/annotations/instances_val2014.json"  # ★ 新加

    # ─── 输出目录 ─────────────────────────────────────────────────
    output_dir:     str = "/root/autodl-tmp/wh5319/silip_clip_caption/image_captioning_final/outputs"
    features_dir:   str = "/root/autodl-tmp/wh5319/silip_clip_caption/image_captioning_final/outputs/features"
    vaw_index_dir:  str = "/root/autodl-tmp/wh5319/silip_clip_caption/attribute_index_vaw_enhanced_siglip2"
    dual_index_dir: str = "/root/autodl-tmp/wh5319/silip_clip_caption/dual_index_final"

    # ─── PKL 检索库 (不变) ────────────────────────────────────────
    name_pkl: str = (
        "/root/autodl-tmp/wh5319/silip_clip_caption/"
        "image_captioning_final/ext_memory_lvis.pkl")
    attr_pkl: str = (
        "/root/autodl-tmp/wh5319/silip_clip_caption/"
        "image_captioning_final/attr_memory/attr_memory.pkl")
    attr_map: str = (
        "/root/autodl-tmp/wh5319/silip_clip_caption/"
        "image_captioning_final/attr_memory/name2attr_topk.json")

    _checkpoint_dir: Optional[str] = None
    _log_dir:        Optional[str] = None

    @property
    def checkpoint_dir(self) -> str:
        return self._checkpoint_dir or os.path.join(self.output_dir, "checkpoints")

    @checkpoint_dir.setter
    def checkpoint_dir(self, value):
        self._checkpoint_dir = value

    @property
    def log_dir(self) -> str:
        return self._log_dir or os.path.join(self.output_dir, "logs")

    @log_dir.setter
    def log_dir(self, value):
        self._log_dir = value

    # [v6.2] LAVIS eva_vit_g.pth (BLIP-2 官方视觉编码器)
    eva_vit_g_model: str = (
        "/root/autodl-tmp/wh5319/silip_clip_caption/"
        "image_captioning_final/eva_vit_g.pth")

    # [v6.2] 兼容字段
    eva01_model: str = (
        "/root/autodl-tmp/wh5319/silip_clip_caption/"
        "image_captioning_final/eva_vit_g.pth")
    eva01_arch: str = "LAVIS-EVA-ViT-g-14"

    # ─── ★ [V6.5] Helper: 按图片路径前缀分发 instances JSON ───────
    # train.py 里 CocoBoxLoader 这样用:
    #     ins_path = cfg.paths.resolve_instances_json(image_rel)
    def resolve_instances_json(self, image_rel: str) -> str:
        """根据图片相对路径(如 'val2014/xxx.jpg')返回对应 instances JSON 路径"""
        if image_rel.startswith('val2014/') or 'COCO_val2014_' in image_rel:
            return self.lvis_val_json
        return self.lvis_train_json


@dataclass
class ClipConfig:
    input_resolution: int  = 224
    embed_dim: int         = 1408
    output_dim: int        = 1408
    image_size: int        = 224
    patch_size: int        = 14
    patch_grid_src: int    = 16
    patch_grid_tgt: int    = 8
    freeze: bool           = True
    daco_enabled: bool     = False
    daco_target_tokens: int = 256
    daco_method: str       = "learned_pooling"
    daco_num_heads: int    = 8
    daco_num_layers: int   = 1


@dataclass
class EvaConfig:
    """[LEGACY-RENAMED] 原名 SiglipConfig. 实际模型跑 EVA-ViT-g/14
    (224x224, patch=14, grid=16). 本 dataclass 里 image_size=512 /
    patch_size=16 这些字段是历史遗留, 模型代码并不读取它们.

    真正还在用的字段只有:
      - mid_layer_offset:  EVA 中层 patch 抓取的层数偏移
      - ms_init_scale:     多尺度融合的初始 scale
    其余字段保留是为了不破坏旧 yaml / ckpt 兼容性 ("不要删除" 约束).
    """
    input_resolution: int  = 512
    embed_dim: int         = 768
    image_size: int        = 512
    patch_size: int        = 16
    patch_grid_src: int    = 32
    patch_grid_tgt: int    = 16
    roi_coord_max_patches: int = 25
    freeze_stage1: bool    = True
    finetune_stage2: bool  = True
    daco_enabled: bool     = False
    daco_target_tokens: int = 1024
    daco_method: str       = "learned_pooling"
    daco_num_heads: int    = 8
    daco_num_layers: int   = 1
    mid_layer_offset: int  = 9
    ms_init_scale: float   = 0.3


# [BACKWARD-COMPAT] 旧名别名, 防外部代码 import SiglipConfig 报错
SiglipConfig = EvaConfig


@dataclass
class SpatialConfig:
    absolute_dim: int          = 10
    relative_dim: int          = 12
    hidden_dim: int            = 256
    output_dim: int            = 768
    use_qformer: bool          = True
    qformer_queries: int       = 32
    qformer_heads: int         = 12
    qformer_layers: int        = 12
    clip_tower_queries: int    = 32
    eva_tower_queries: int     = 32   # [LEGACY-RENAMED] 原 siglip_tower_queries
    spatial_tower_queries: int = 18
    roi_detail_queries: int    = 32
    global_stream_queries: int = 32
    object_stream_queries: int = 32
    # [v6.3 继承] ALL-CORE: 9 个 ROI 全部当核心
    roi_select_num: int        = 9
    roi_per_roi_queries: int   = 32
    roi_align_output_size: int = 14
    crossvit_bridge_layers: int = 2
    dual_stream_crossvit_layers: int = 2
    llip_num_mixture: int      = 4


@dataclass
class AttributeConfig:
    similarity_threshold: float = 0.27
    embed_dim: int              = 768
    max_attrs_per_roi: int      = 4


@dataclass
class DecoderConfig:
    vocab_size: int  = 32000
    embed_dim: int   = 768
    llm_hidden: int  = 4096
    max_length: int  = 64
    dropout: float   = 0.1


@dataclass
class LoraConfig:
    enabled: bool   = False
    r: int          = 16
    lora_alpha: int = 32
    target_modules: list = field(default_factory=lambda: ["q_proj", "v_proj"])
    dropout: float  = 0.05


@dataclass
class LossConfig:
    ntp_weight: float      = 1.0
    label_smoothing: float = 0.00
    w_sigmoid: float       = 0.0
    w_roi: float           = 0.0


@dataclass
class TrainingConfig:
    total_epochs: int                = 3
    stage1_epochs: int               = 0
    stage1_lr: float                 = 1e-4
    stage2_epochs: int               = 3
    stage2_lr: float                 = 3e-5
    batch_size: int                  = 4
    gradient_accumulation_steps: int = 4
    # [v6.3 继承] max_rois=9, core_rois=9
    max_rois: int                    = 9
    core_rois: int                   = 9
    weight_decay: float              = 0.01
    gradient_clip: float             = 1.0
    warmup_ratio: float              = 0.05
    device: str                      = "cuda"
    use_amp: bool                    = True
    num_workers: int                 = 0
    save_steps: int                  = 1000
    eval_steps: int                  = 500
    logging_steps: int               = 100
    num_captions: int                = 5
    print_samples_every: int         = 200
    mode: str                        = "online"
    max_len: int                     = 64
    scst_epochs: int                 = 0
    scst_lr: float                   = 5e-6
    w_contrast: float                = 0.0
    contrast_warmup_steps: int       = 500
    w_ret_align: float               = 0.0
    ret_align_warmup_epochs: int     = 0
    unfreeze_new_modules_epoch: int  = 99
    soft_retriever_warmup_epochs: int = 0
    soft_retriever_warmup_lr: float  = 3e-4
    ls_curriculum_epoch: int         = 0
    # ★ FAST IO (v6.4) ★ 让 yaml 的 dataloader_num_workers 等字段能被读到
    dataloader_num_workers: int      = 16
    persistent_workers: bool         = True
    prefetch_factor: int             = 4
    pin_memory: bool                 = True


@dataclass
class AblationConfig:
    use_attribute: bool        = True
    use_spatial: bool          = True
    use_distill: bool          = False
    use_qformer: bool          = True
    use_tag_anchor: bool       = True
    use_dual_index: bool       = True
    use_aoa_gate: bool         = True
    use_ms_eva: bool           = True   # [LEGACY-RENAMED] 原 use_ms_siglip, 控制 EVA 中层 patch 多尺度融合
    ablation_name: str         = "full"


@dataclass
class ExperimentConfig:
    # ★ [v6.5] 实验名 → v6_5_karpathy118k (ckpt 跟旧 v6.4 物理分开)
    name: str = "caption_v6_5_karpathy118k"
    seed: int = 42
    resume_from_checkpoint: Optional[str] = None


@dataclass
class V4PipelineConfig:
    """[v6.3 继承] 与 GlobalROIBidirectionalFusion 默认值同步 (max_core=9, max_non_core=0)"""
    core_roi_count: int    = 9
    noncore_roi_count: int = 0


@dataclass
class Config:
    preprocess:   PreprocessConfig   = field(default_factory=PreprocessConfig)
    paths:        PathsConfig        = field(default_factory=PathsConfig)
    clip:         ClipConfig         = field(default_factory=ClipConfig)
    eva:          EvaConfig          = field(default_factory=EvaConfig)   # [LEGACY-RENAMED] 原 siglip
    spatial:      SpatialConfig      = field(default_factory=SpatialConfig)
    attribute:    AttributeConfig    = field(default_factory=AttributeConfig)
    decoder:      DecoderConfig      = field(default_factory=DecoderConfig)
    lora:         LoraConfig         = field(default_factory=LoraConfig)
    loss:         LossConfig         = field(default_factory=LossConfig)
    training:     TrainingConfig     = field(default_factory=TrainingConfig)
    ablation:     AblationConfig     = field(default_factory=AblationConfig)
    experiment:   ExperimentConfig   = field(default_factory=ExperimentConfig)
    v4_pipeline:  V4PipelineConfig   = field(default_factory=V4PipelineConfig)

    @property
    def experiment_name(self) -> str:
        return f"{self.experiment.name}_{self.ablation.ablation_name}"

    @property
    def llip_num_mixture(self) -> int:
        return self.spatial.llip_num_mixture


def _find_config_yaml() -> Optional[str]:
    for p in [
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml"),
        os.path.join(os.getcwd(), "config.yaml"),
        "/root/autodl-tmp/wh5319/silip_clip_caption/image_captioning_final/config.yaml",
    ]:
        if os.path.exists(p):
            return p
    return None


def _migrate_legacy_yaml_keys(raw: dict) -> dict:
    """[LEGACY-YAML-COMPAT] 旧 yaml 里 siglip 命名自动迁移到新名.
    新 yaml 应该写 eva:/use_ms_eva:/eva_tower_queries:, 但旧 yaml 仍然能跑.
    """
    if not isinstance(raw, dict):
        return raw
    migrated = 0
    # 顶层段名: siglip → eva (如果两者都存在以 eva 优先)
    if 'siglip' in raw and 'eva' not in raw:
        raw['eva'] = raw.pop('siglip')
        migrated += 1
    elif 'siglip' in raw:
        raw.pop('siglip')  # eva 已存在, 丢弃 siglip
        migrated += 1
    # paths.siglip_model → paths._unused_siglip_model_path
    if isinstance(raw.get('paths'), dict) and 'siglip_model' in raw['paths']:
        raw['paths']['_unused_siglip_model_path'] = raw['paths'].pop('siglip_model')
        migrated += 1
    # ablation.use_ms_siglip → use_ms_eva
    if isinstance(raw.get('ablation'), dict) and 'use_ms_siglip' in raw['ablation']:
        raw['ablation']['use_ms_eva'] = raw['ablation'].pop('use_ms_siglip')
        migrated += 1
    # spatial.siglip_tower_queries → eva_tower_queries
    if isinstance(raw.get('spatial'), dict) and 'siglip_tower_queries' in raw['spatial']:
        raw['spatial']['eva_tower_queries'] = raw['spatial'].pop('siglip_tower_queries')
        migrated += 1
    if migrated > 0:
        print(f"[config] [LEGACY-YAML] 自动迁移 {migrated} 个 siglip_* 旧 key 到新名")
    return raw


def get_config(yaml_path: Optional[str] = None) -> Config:
    if yaml_path is None:
        yaml_path = _find_config_yaml()
    cfg = Config()
    if yaml_path is None or not os.path.exists(yaml_path):
        print("[config] 未找到 config.yaml，使用默认配置")
        return cfg
    print(f"[config] 加载: {yaml_path}")
    with open(yaml_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if not raw:
        return cfg
    raw = _migrate_legacy_yaml_keys(raw)
    for section in [
        'preprocess', 'paths', 'clip', 'eva', 'spatial',
        'attribute', 'decoder', 'lora', 'loss', 'training',
        'ablation', 'experiment', 'v4_pipeline',
    ]:
        if section in raw and hasattr(cfg, section):
            _apply_dict(getattr(cfg, section), raw.get(section))
    return cfg


ABLATION_PRESETS = {
    "baseline": dict(
        use_attribute=False, use_spatial=False, use_dual_index=False,
        use_qformer=False,   use_tag_anchor=False,
        use_aoa_gate=False,  use_ms_eva=False,
        ablation_name="baseline"),
    "+attr": dict(
        use_attribute=True,  use_spatial=False, use_dual_index=False,
        use_qformer=False,   use_tag_anchor=True,
        use_aoa_gate=False,  use_ms_eva=False,
        ablation_name="attr"),
    "+spatial": dict(
        use_attribute=False, use_spatial=True,  use_dual_index=False,
        use_qformer=True,    use_tag_anchor=False,
        use_aoa_gate=False,  use_ms_eva=False,
        ablation_name="spatial"),
    "+dual": dict(
        use_attribute=True,  use_spatial=True,  use_dual_index=True,
        use_qformer=True,    use_tag_anchor=True,
        use_aoa_gate=False,  use_ms_eva=False,
        ablation_name="dual"),
    "+aoa": dict(
        use_attribute=True,  use_spatial=True,  use_dual_index=True,
        use_qformer=True,    use_tag_anchor=True,
        use_aoa_gate=True,   use_ms_eva=False,
        ablation_name="aoa"),
    "+ms": dict(
        use_attribute=True,  use_spatial=True,  use_dual_index=True,
        use_qformer=True,    use_tag_anchor=True,
        use_aoa_gate=False,  use_ms_eva=True,
        ablation_name="ms"),
    "full": dict(
        use_attribute=True,  use_spatial=True,  use_dual_index=True,
        use_qformer=True,    use_tag_anchor=True,
        use_aoa_gate=True,   use_ms_eva=True,
        ablation_name="full"),
}


def get_ablation_config(
    ablation_name: str = "full",
    yaml_path: Optional[str] = None,
) -> Config:
    cfg    = get_config(yaml_path)
    preset = ABLATION_PRESETS.get(ablation_name, ABLATION_PRESETS["full"])
    if ablation_name not in ABLATION_PRESETS:
        print(f"[config] 未知消融 '{ablation_name}'，使用 full")
    _apply_dict(cfg.ablation, preset)
    print(f"[config] 消融方案: {ablation_name}")
    return cfg


def create_directories(config: Config):
    for d in [
        config.paths.output_dir,
        config.paths.checkpoint_dir,
        config.paths.log_dir,
        os.path.join(config.paths.features_dir, "train"),
        os.path.join(config.paths.features_dir, "val"),
    ]:
        os.makedirs(d, exist_ok=True)


def print_config(config: Config):
    ab  = config.ablation
    eva = config.eva
    tr  = config.training
    pp  = config.preprocess
    p   = config.paths

    print(f"\n{'='*72}")
    print(f"  {config.experiment_name}")
    print(f"  V6.5: Karpathy 118k 适配 (基于 v6.4 A+B 双修复)")
    print(f"{'='*72}")

    # ─── ★ V6.5 数据集信息 (新加) ───────────────────────────────
    print(f"  [V6.5 NEW] 数据集切换: Karpathy 118k split")
    print(f"")
    print(f"    dataset_format:   {getattr(p, 'dataset_format', 'karpathy')}")
    if getattr(p, 'dataset_format', 'karpathy') == 'karpathy':
        print(f"    train JSON:       {p.karpathy_train_json}")
        print(f"    val JSON:         {p.karpathy_val_json}")
        print(f"    test JSON:        {p.karpathy_test_json}")
        print(f"    image_root:       {p.karpathy_image_root}")
        print(f"                      ↑ 父目录, 必须含 train2014/ + val2014/ 两个子目录")
        print(f"    格式: 扁平 [{{'caption','image','image_id'}}, ...] (1 caption / record)")
    else:
        print(f"    train caption:    {p.coco_train_captions}")
        print(f"    train images:     {p.coco_train_images}")
        print(f"    格式: COCO 原版 {{'images':[..], 'annotations':[..]}}")
    print(f"")
    print(f"    [DUAL-INSTANCES] ★ Karpathy train.json 同时含 train2014/ + val2014/")
    print(f"    train instances:  {p.lvis_train_json}")
    print(f"    val instances:    {getattr(p, 'lvis_val_json', '(未配置)')}")
    print(f"                      ↑ CocoBoxLoader 按图片前缀自动分发到对应 instances")

    print(f"  [视觉编码器] LAVIS 对齐版 (与 v6.2 一致)")
    print(f"    模型:      LAVIS EVA-ViT-g/14 (BLIP-2 官方)")
    print(f"    权重:      {getattr(p, 'eva_vit_g_model', p.eva01_model)}")
    print(f"    架构:      depth=39, embed_dim=1408, patch=14, img_size=224")
    print(f"    预处理:    Resize((224,224)) BICUBIC + CLIP mean/std")

    print(f"  [v6.4 继承] A + B 双修复")
    print(f"")
    print(f"    ── B 修复: Q-Former 内部分流 ─────────────────────────")
    print(f"    [TYPE-EMB]    evcap_qformer.query_type_emb = nn.Embedding(2, 768)")
    print(f"                  - 0=name query, 1=attr query")
    print(f"                  - 加在 query_in_proj 之后, 12 层 self-attn 起就分化")
    print(f"                  - std=0.02 初始化, 不会压制预训练 BLIP2 信号")
    print(f"    [Q-PAD]       forward_with_queries 支持 query_padding_mask")
    print(f"                  - True=有效, False=padding (self-attn 屏蔽 -1e4)")
    print(f"                  - 通过 _LAVISBertLayer.self_attn_mask 透传")
    print(f"")
    print(f"    ── A 修复: LLM 端 prompt 分段 ────────────────────────")
    print(f"    [LLM-SEG]     MistralDecoder 接受 (name_tokens_768, attr_tokens_768)")
    print(f"                  - _OBJECTS_MARKER    = '\\nObjects: '")
    print(f"                  - _ATTRIBUTES_MARKER = '\\nAttributes: '")
    print(f"                  v6.2: <BOS><SYS> <vis_32> <name_8>             <INSTRUCT>")
    print(f"                  v6.4: <BOS><SYS> <vis_32> Objects:<name_8>")
    print(f"                                            Attributes:<attr_16> <INSTRUCT>")
    print(f"")
    print(f"    ── 桥接: CaptionModel ───────────────────────────────")
    print(f"    [SPLIT-NA]    _collect_name_tokens_evcap 返回 3-tuple")
    print(f"                  (name_tokens [B,8,768], attr_tokens [B,16,768], names)")
    print(f"                  K_NAME_SLOTS=8 + K_ATTR_SLOTS=16 (硬编码于 model.py 顶部)")
    print(f"                  - 拼接 [name|attr] 共 24 query, 一次 forward")
    print(f"                  - query_types: 前 8 是 0(name), 后 16 是 1(attr)")
    print(f"                  - query_padding_mask: 标记实际填充槽位")
    print(f"                  - 输出按 [:8] / [8:24] 拆回 name/attr")
    print(f"    [STAGE2]      unfreeze_for_stage2 / _lite / freeze_for_soft_stage1")
    print(f"                  全部同步解冻 evcap_qformer.query_type_emb")
    print(f"    [SHARED-PROJ] name_embeds + attr_embeds 共享 vis_proj_A")

    print(f"  [per-query 检索] 32 个 query token 独立 top-K → 合并排序去重")
    print(f"    第1步: retrieval_qformer(patch) → global_32q [B, 32, 768]")
    print(f"    第2步: 每个 query 独立 top-5 → 32×5 候选 → sort → dedupe → top-K=8")
    print(f"    第3步: name+attr 拼接定长 [K_n=8 | K_a=16] 一次 padded forward")
    print(f"           带 type bias + padding mask, 输出拆回两路")

    print(f"  [ROI Pipeline]  max_rois={pp.max_rois}")
    print(f"    ROIAlignPooler(49tok) + PerROIQFormer + ROITokenAggregator")
    print(f"    ROINameAttrInjector: 属性注入 ROI 特征")

    print(f"  [TriStreamFusion]  视觉融合统一通道")
    print(f"    global_32q (LAVIS 产) + vis_32q (ROI 流) + cat(name_8, attr_16)")

    print(f"  [PKL 双记忆库]  同一 768 维 LAVIS 空间")
    print(f"    名称库: {p.name_pkl}")
    print(f"    属性库: {p.attr_pkl}")
    print(f"    桥接:   {p.attr_map}")

    print(f"  [训练] train.py + Karpathy dataset")
    print(f"    [5GT-EXPAND] Karpathy 扁平格式天然每图 5 条 record (无需展开)")
    print(f"    Stage2:     {tr.stage2_epochs}ep  lr={tr.stage2_lr:.2e}")
    print(f"    max_rois:   {tr.max_rois}  core_rois: {tr.core_rois}")
    print(f"    LS-CUR:     ls_curriculum_epoch={tr.ls_curriculum_epoch}  "
          f"(0=不启用)")

    print(f"  [损失函数] 仅 NTP (无辅助 loss, losses.py 无需改动)")
    print(f"    NTP loss:   label_smoothing={config.loss.label_smoothing}")
    print(f"    Contrast:   禁用 (w_contrast={tr.w_contrast})")
    print(f"    RetAlign:   禁用 (w_ret_align={tr.w_ret_align})")

    print(f"  [LLM]     {p.llm_path}")
    print(f"  [训练设置] Stage2={tr.stage2_epochs}轮 "
          f"batch={tr.batch_size}×accum={tr.gradient_accumulation_steps}")
    print(f"  [多尺度]  use_ms_eva={ab.use_ms_eva}  "
          f"mid_layer_offset={eva.mid_layer_offset}")
    print(f"{'='*72}\n")