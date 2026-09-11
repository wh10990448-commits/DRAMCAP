#!/usr/bin/env python3
"""
config.py  v6.6-dual  [SIGLIP-LEGACY-RENAMED]
==============================================
v6.6 双卡 32GB 配置加载器。

提供给 train.py 的接口 (跟单卡版完全一致):
  - get_config(yaml_path=None)              → 返回完整 SimpleNamespace 配置
  - get_ablation_config(name='full', ...)   → 在 'full' 基础上 toggle 特性
  - create_directories(config)              → mkdir 输出 / checkpoint / log 目录
  - print_config(config)                    → 打印关键配置 (rank 0 调用)

[SIGLIP-LEGACY-RENAMED] 改动:
  - cfg.ablation.use_ms_siglip → cfg.ablation.use_ms_eva
  - ablation 子命令 'no_ms_siglip' → 'no_ms_eva'
  - print_config 显示 'ms_eva' 而非 'ms_siglip'
  - 旧 yaml (写 'use_ms_siglip: true' 的) 自动迁移到 use_ms_eva
  - 旧 ablation 名 'no_ms_siglip' 仍然兼容 (alias 到 'no_ms_eva')
"""

import os
from types import SimpleNamespace

try:
    import yaml
except ImportError as e:
    raise ImportError("需要 pyyaml: pip install pyyaml") from e


_DEFAULT_YAML = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    'config.yaml'
)


# ============================================================
# 内部工具: dict → SimpleNamespace (支持 cfg.foo.bar 风格)
# ============================================================
def _dict_to_ns(d):
    if isinstance(d, dict):
        return SimpleNamespace(**{k: _dict_to_ns(v) for k, v in d.items()})
    if isinstance(d, list):
        return [_dict_to_ns(x) for x in d]
    return d


def _load_yaml(path=None):
    p = path or _DEFAULT_YAML
    if not os.path.exists(p):
        raise FileNotFoundError(f"config.yaml 不存在: {p}")
    with open(p, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def _migrate_legacy_yaml_keys(raw):
    """[LEGACY-YAML-COMPAT] 旧 yaml 里 siglip_* 命名自动迁移到新名.

    跟 config_loader.py 行为完全一致, 保证两个加载器互通.
    """
    if not isinstance(raw, dict):
        return raw
    migrated = 0
    # 顶层段名: siglip → eva (如两者都存在以 eva 优先)
    if 'siglip' in raw and 'eva' not in raw:
        raw['eva'] = raw.pop('siglip')
        migrated += 1
    elif 'siglip' in raw:
        raw.pop('siglip')
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


# ============================================================
# 对外: get_config / get_ablation_config
# ============================================================
def get_config(yaml_path=None):
    """加载完整配置"""
    raw = _load_yaml(yaml_path)
    raw = _migrate_legacy_yaml_keys(raw)
    cfg = _dict_to_ns(raw)
    cfg.experiment_name = getattr(getattr(cfg, 'experiment', None), 'name', 'default')
    return cfg


def get_ablation_config(name='full', yaml_path=None):
    """获取消融配置 (在 'full' 基础上 toggle 某些特性).

    支持的 name:
      - 'full'           不变
      - 'no_spatial'     关空间特征
      - 'no_attribute'   关属性
      - 'no_ms_eva'      关 EVA 多尺度融合  [SIGLIP-LEGACY-RENAMED]
      - 'no_ms_siglip'   旧名, 自动 alias 到 'no_ms_eva' (向后兼容)
    """
    cfg = get_config(yaml_path)
    base = cfg.experiment_name
    # 旧 ablation 名兼容
    if name == 'no_ms_siglip':
        print(f"[config] [LEGACY] ablation 'no_ms_siglip' → 'no_ms_eva'")
        name = 'no_ms_eva'

    if name == 'full':
        pass
    elif name == 'no_spatial':
        cfg.ablation.use_spatial = False
        cfg.experiment_name = f"{base}_no_spatial"
    elif name == 'no_attribute':
        cfg.ablation.use_attribute = False
        cfg.experiment_name = f"{base}_no_attribute"
    elif name == 'no_ms_eva':
        cfg.ablation.use_ms_eva = False
        cfg.experiment_name = f"{base}_no_ms_eva"
    else:
        print(f"[config] 未知 ablation '{name}', 使用 'full'")
    return cfg


# ============================================================
# 对外: 工具
# ============================================================
def create_directories(config):
    """mkdir 输出 / checkpoint / log 目录"""
    for attr in ('output_dir', 'checkpoint_dir', 'log_dir'):
        p = getattr(config.paths, attr, None)
        if p:
            os.makedirs(p, exist_ok=True)


def print_config(config):
    """打印关键配置 (只在 rank 0 调用)"""
    sep = "=" * 70
    print(sep)
    print(f"  Experiment: {getattr(config, 'experiment_name', '?')}")
    print(sep)
    t = config.training
    bs_per_gpu = t.batch_size
    ga = t.gradient_accumulation_steps
    # 双卡时全局 batch 由 train.py 实际填入 WORLD_SIZE 决定; 这里只显示 per-GPU
    print(f"  [Training] batch_size/gpu={bs_per_gpu}  grad_accum={ga}  "
          f"effective_batch/step = bs * grad_accum * world_size")
    print(f"  [Training] stage2_epochs={t.stage2_epochs}  lr={t.stage2_lr}  "
          f"weight_decay={t.weight_decay}")
    print(f"  [Training] num_captions_per_image={t.num_captions_per_image}  "
          f"max_len={t.max_len}  max_rois={t.max_rois}")
    print(f"  [DataLoader] num_workers={t.dataloader_num_workers}  "
          f"prefetch_factor={t.prefetch_factor}  "
          f"persistent_workers={t.persistent_workers}")
    print(f"  [Paths] llm:  {config.paths.llm_path}")
    print(f"  [Paths] vis:  {config.paths.eva_vit_g_model}")
    print(f"  [Paths] ckpt: {config.paths.checkpoint_dir}")
    print(f"  [Paths] log:  {config.paths.log_dir}")
    # [SIGLIP-LEGACY-RENAMED] use_ms_siglip → use_ms_eva
    # 仍读 use_ms_siglip 作 fallback, 兼容外部代码可能传入旧名 ablation 对象
    use_ms_eva = getattr(config.ablation, 'use_ms_eva',
                         getattr(config.ablation, 'use_ms_siglip', True))
    print(f"  [Ablation] spatial={config.ablation.use_spatial}  "
          f"attribute={config.ablation.use_attribute}  "
          f"ms_eva={use_ms_eva}")
    print(sep)


# 向后兼容: 旧 run.sh 检查 config_loader.py 的存在
# (我们把功能全合并到 config.py, 提供这个别名)
_load = _load_yaml