#!/bin/bash
# ============================================================================
# 图像描述生成系统  v6.7 (Karpathy 118k + 双 instances 合并)
#                  继承 v6.6 全部 (ROI 受限 attr 检索 + v6.8 attr_memory + v6.9.3 retriever)
# caption_model 包结构 + train.py v6.1 + losses.py v6.0 + config v6.5
#
# v6.7 改动 (相对 v6.6): ★ 仅 3 处, 其他字节级一致
#   [LOAD-CFG]      load_config() 加 2 个 yaml_get:
#                     coco_karpathy_train_json → KARPATHY_TRAIN_JSON
#                     lvis_val_json             → COCO_INSTANCES_VAL
#   [GT-STATS-FMT]  print_gt_stats() 用 isinstance(data, list) 自动识别格式:
#                     list  → Karpathy 扁平 (每条 = 1 caption)
#                     dict  → COCO 原版 ({images, annotations})
#   [CMD-ARG]       train_online() 在 python train.py 命令末尾加:
#                     --train_json $KARPATHY_TRAIN_JSON  (Karpathy 训练 JSON)
#                     --lvis_val_json $COCO_INSTANCES_VAL (val instances, 双合并)
#
# v6.6 继承 (相对 v6.5+/v6.8-attr-typed, 全部不变):
#   ── attr 检索改 ROI 受限 (model.py + retriever.py) ───────────
#   [V6.6-ATTR]     ★ Attr 检索: global query → ROI feature + 受限检索
#                   - 用 retrieval_vecs[b] (per-ROI 768d) 作 attr 检索 query
#                   - 只在该 ROI 物体的 attr 子集里检索 (绕过库内塌缩)
#                   - 实测 mean sim 0.514 → 0.759 (+0.245)
#                   - 配对错位修对 (motorcycle 'red' 不会跑到 person 上)
#   [V6.9.3-RETR]   retriever.py 加 retrieve_attrs_per_roi_constrained
#                   + COCO_TO_ATTR_ALIAS 同义词 (couch→sofa, donut→doughnut)
#                   + _resolve_obj_alias 自动应用 (旧路径也受益)
#   [V6.6-COMPAT]   完全向后兼容旧 ckpt:
#                   - 没新增/删除任何 nn.Module
#                   - 旧 ckpt strict=False 直接加载, 只是 attr 行为变了
#   [V6.6-FALLBACK] 推理时若无 ROI / 全不命中, 自动降级走 v6.5 global
#   [V6.6-CKPT]     checkpoint_dir / log_dir 加 _v66_roi 后缀
#                   → v6.5 ep1=1.355 完整保留 (在 outputs/checkpoints/)
#                   → v6.6 训练写入 outputs/checkpoints_v66_roi/
#
# v6.8 继承 (attr_memory.pkl 不变):
#   [V6.8-DEP]      vg_attribute_types.json (classify_vg_attrs_v2.py 生成)
#   [V6.8-MERGE]    VAW + VG 合并为 unified attr→dim 表 (~94 dim)
#   [V6.8-LVIS]     LVIS 覆盖逻辑跟 v6.5+/v6.7 完全一致
#
# v6.5+ 继承 (不变):
#   [V6.5+-ATTR]    PACO + VAW + VG 三源融合
#   [V6.5+-VERIFY]  verify 子命令: 验证三方对齐
#   [V6.5+-BUILD]   build_attr 子命令: 重建 attr_memory.pkl
#
# v6.4 继承 (不变):
#   [V6.4-PKG]      caption_model/ 包结构 (16 文件)
#   [V6.4-LAVIS]    Q-Former = EVCap 原版 LAVIS 实现
#
# v6.3 视觉/检索 (不变):
#   [LAVIS-VIS]     LAVIS eva_vit_g.pth + Resize(224,224) BICUBIC + CLIP
#   [PER-Q]         per-query-token 检索 (32q × top-5 → 合并去重) — name 用
#   [COSINE]        同源 768 维空间 (name + attr 全在 BLIP-2 768 空间)
#   [TriStream]     global_32q + vis_32q + cat(name+attr)
#   [LOSS]          仅 NTP cross-entropy
#   [5GT-EXPAND]    每图 × 5 条 GT 展开
# ============================================================================
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="${SCRIPT_DIR}/config.yaml"

find_python() {
    if python3 -c "import torch" 2>/dev/null; then echo "python3"; return 0; fi
    for d in /root/miniconda3/envs/*/bin /opt/conda/envs/*/bin /root/anaconda3/envs/*/bin; do
        [ -f "${d}/python" ] || continue
        if "${d}/python" -c "import torch" 2>/dev/null; then echo "${d}/python"; return 0; fi
    done
    for p in /root/miniconda3/bin/python /opt/conda/bin/python /root/anaconda3/bin/python; do
        [ -f "$p" ] || continue
        if "$p" -c "import torch" 2>/dev/null; then echo "$p"; return 0; fi
    done
    return 1
}

setup_python() {
    if [ -n "$PYTHON" ]; then
        if ! "$PYTHON" -c "import torch" 2>/dev/null; then
            echo -e "\033[0;31m❌ 指定的 PYTHON=$PYTHON 没有安装 torch\033[0m" >&2
            exit 1
        fi
        return 0
    fi
    PYTHON=$(find_python) || {
        echo -e "\033[0;31m❌ 未找到安装了 PyTorch 的 Python\033[0m" >&2
        echo "请先激活含 PyTorch 的环境: conda activate <your_env>"
        exit 1
    }
    echo -e "\033[0;32m✅ Python: ${PYTHON}\033[0m"
    $PYTHON -c "
import torch
v = torch.__version__; c = torch.cuda.is_available()
print(f'   PyTorch {v}  CUDA={c}')
if c:
    p = torch.cuda.get_device_properties(0)
    print(f'   GPU: {p.name}  显存: {p.total_memory/1e9:.1f}GB')
" 2>/dev/null || true
}

yaml_get() {
    local file="$1" section="$2" key="$3"
    [ ! -f "$file" ] && echo "" && return
    $PYTHON - "$file" "$section" "$key" << 'PYEOF'
import sys, re
file, section, key = sys.argv[1], sys.argv[2], sys.argv[3]
with open(file) as f: lines = f.readlines()
in_section = False
for line in lines:
    stripped = line.rstrip()
    if re.match(r'^\s*#', stripped) or not stripped.strip(): continue
    m_sec = re.match(r'^([a-zA-Z_][a-zA-Z0-9_]*):\s*$', stripped)
    if m_sec: in_section = (m_sec.group(1) == section); continue
    if in_section:
        m_val = re.match(r'^\s+' + re.escape(key) + r':\s*(.+)', stripped)
        if m_val:
            val = m_val.group(1).strip()
            val = re.sub(r'\s+#.*$', '', val)
            val = val.strip().strip('"').strip("'")
            print(val); sys.exit(0)
PYEOF
}

_find_qformer_weight() {
    local candidates=("$QFORMER_WEIGHT" \
        "/root/autodl-tmp/wh5319/silip_clip_caption/model/blip2_pretrained_flant5xxl.pth")
    for p in "${candidates[@]}"; do
        [ -n "$p" ] && [ -f "$p" ] && QFORMER_WEIGHT="$p" && return 0
    done
    QFORMER_WEIGHT=""
}

load_config() {
    echo "📄 加载配置: ${CONFIG_FILE}"

    _v="$(yaml_get "$CONFIG_FILE" paths eva_vit_g_model)";      VIS_ENCODER="${_v}"
    if [ -z "$VIS_ENCODER" ]; then
        _v="$(yaml_get "$CONFIG_FILE" paths eva01_model)";      VIS_ENCODER="${_v}"
    fi
    _v="$(yaml_get "$CONFIG_FILE" paths eva01_arch)";           VIS_ARCH="${_v:-LAVIS-EVA-ViT-g-14}"

    _v="$(yaml_get "$CONFIG_FILE" paths llm_path)";             LLM_PATH="${_v}"
    _v="$(yaml_get "$CONFIG_FILE" paths blip2_qformer_path)";   QFORMER_WEIGHT="${_v}"
    _v="$(yaml_get "$CONFIG_FILE" paths coco_train_images)";    COCO_IMAGES="${_v}"
    _v="$(yaml_get "$CONFIG_FILE" paths coco_train_captions)";  COCO_CAPTIONS="${_v}"
    _v="$(yaml_get "$CONFIG_FILE" paths lvis_train_json)";      COCO_INSTANCES="${_v}"
    _v="$(yaml_get "$CONFIG_FILE" paths output_dir)";           OUTPUT_DIR="${_v}"
    _v="$(yaml_get "$CONFIG_FILE" paths checkpoint_dir)";       CKPT_DIR="${_v}"
    _v="$(yaml_get "$CONFIG_FILE" paths log_dir)";              LOG_DIR="${_v}"
    _v="$(yaml_get "$CONFIG_FILE" paths name_pkl)";             NAME_PKL="${_v}"
    _v="$(yaml_get "$CONFIG_FILE" paths attr_pkl)";             ATTR_PKL="${_v}"
    _v="$(yaml_get "$CONFIG_FILE" paths attr_map)";             ATTR_MAP="${_v}"

    # ★★★ [V6.7 NEW 补丁 1/3] Karpathy 训练 JSON + val instances ★★★
    _v="$(yaml_get "$CONFIG_FILE" paths coco_karpathy_train_json)"; KARPATHY_TRAIN_JSON="${_v}"
    _v="$(yaml_get "$CONFIG_FILE" paths lvis_val_json)";            COCO_INSTANCES_VAL="${_v}"

    _v="$(yaml_get "$CONFIG_FILE" training batch_size)";        BATCH_SIZE="${_v:-4}"
    _v="$(yaml_get "$CONFIG_FILE" training mode)";              TRAIN_MODE="${_v:-online}"
    _v="$(yaml_get "$CONFIG_FILE" training stage2_epochs)";     STAGE2_EPOCHS="${_v:-3}"
    _v="$(yaml_get "$CONFIG_FILE" training gradient_accumulation_steps)"; GRAD_ACCUM="${_v:-1}"
    _v="$(yaml_get "$CONFIG_FILE" training ls_curriculum_epoch)"; LS_CURRICULUM="${_v:-0}"
    _v="$(yaml_get "$CONFIG_FILE" training num_captions_per_image)"; NUM_CAPTIONS="${_v:-5}"
    _v="$(yaml_get "$CONFIG_FILE" clip embed_dim)";             CLIP_DIM="${_v:-1408}"
    _v="$(yaml_get "$CONFIG_FILE" clip num_patches)";           CLIP_PATCHES="${_v:-256}"
    _v="$(yaml_get "$CONFIG_FILE" clip image_size)";            CLIP_IMG_SIZE="${_v:-224}"

    TRAIN_SCRIPT="${SCRIPT_DIR}/train.py"
    VERIFY_SCRIPT="${SCRIPT_DIR}/verify_alignment.py"
    BUILD_ATTR_SCRIPT="${SCRIPT_DIR}/build_attr_memory.py"
    CLASSIFY_VG_SCRIPT="${SCRIPT_DIR}/classify_vg_attrs_v2.py"
    VG_ATTR_TYPES_JSON="/root/autodl-tmp/wh5319/all_data/VG/vg_attribute_types.json"

    if [ -z "$QFORMER_WEIGHT" ] || [ ! -e "$QFORMER_WEIGHT" ]; then
        _find_qformer_weight
    fi
}

GREEN='\033[0;32m'; BLUE='\033[0;34m'; YELLOW='\033[1;33m'
RED='\033[0;31m'; CYAN='\033[0;36m'; NC='\033[0m'
header() { echo -e "${GREEN}\n============================================================\n  $1\n============================================================${NC}"; }
step()   { echo -e "${BLUE}>>> $1${NC}"; }
ok()     { echo -e "${GREEN}✅ $1${NC}"; }
warn()   { echo -e "${YELLOW}⚠️  $1${NC}"; }
err()    { echo -e "${RED}❌ $1${NC}" >&2; exit 1; }

# ────────────────────────────────────────────────────────────────────
# [V6.8-CHECK] 增强版 attr_memory.pkl 显示 (v6.6 不变, schema 完全兼容)
# ────────────────────────────────────────────────────────────────────
print_attr_memory_v68() {
    local attr_pkl_path="$1"
    [ -z "$attr_pkl_path" ] || [ ! -f "$attr_pkl_path" ] && return
    $PYTHON - "$attr_pkl_path" << 'PYEOF' 2>/dev/null || true
import sys, pickle, numpy as np
from collections import Counter
attr_pkl = sys.argv[1]
try:
    with open(attr_pkl, 'rb') as f: am = pickle.load(f)
    ak  = am.get('keys')
    n2a = am.get('name2attr_idx', {})
    dims = am.get('dims', [])
    sources = am.get('sources', None)
    coverage = am.get('lvis_coverage', None)
    data_sources = am.get('data_sources', None)
    version = am.get('version', '?')

    if isinstance(ak, np.ndarray):
        print(f"  ✅ 属性库: keys{list(ak.shape)} | {len(n2a)} 物体有属性 | version={version}")
        unique_dims = sorted(set(dims))
        dim_counts = Counter(dims)
        print(f"     维度({len(unique_dims)}): 前 10 → {[d for d,_ in dim_counts.most_common(10)]}")
        print(f"     分布 (前 6): {dict(dim_counts.most_common(6))}")

    if sources:
        src_counter = Counter(sources)
        total = sum(src_counter.values())
        parts = []
        for src in ['PACO', 'VAW', 'VG']:
            cnt = src_counter.get(src, 0)
            if cnt > 0:
                pct = 100 * cnt / total
                parts.append(f"{src}={cnt}({pct:.1f}%)")
        if parts:
            print(f"     三源融合: {' + '.join(parts)}")

    if data_sources:
        srcs_on = [k.upper() for k, v in data_sources.items()
                   if isinstance(v, bool) and v]
        if srcs_on:
            print(f"     启用数据源: {' + '.join(srcs_on)}")

    if coverage:
        cov_total = coverage.get('total_lvis_classes', 0)
        cov_n = coverage.get('covered_lvis_classes', 0)
        cov_rate = coverage.get('coverage_rate', 0)
        coco_cov = coverage.get('coco80_covered', 0)
        print(f"     LVIS 覆盖: {cov_n}/{cov_total} ({cov_rate:.1%})")
        if coco_cov:
            print(f"     COCO 80:  {coco_cov}/80 ({coco_cov/80:.1%})")
        coco_uncov = coverage.get('coco80_uncovered', [])
        if coco_uncov:
            print(f"     COCO 未覆盖: {coco_uncov}")

    n_dims = am.get('n_dims', None)
    n_unified = am.get('n_unified_attr_table', None)
    all_dims_in_lib = am.get('all_dims_in_lib', None)
    v68_changes = am.get('v68_changes', None)

    if n_dims is not None or n_unified is not None or v68_changes:
        print(f"")
        print(f"     ── [V6.8] 标识 ──")
    if n_dims is not None:
        print(f"     入库 dim 数:                {n_dims}")
    if n_unified is not None:
        print(f"     unified attr→dim 表大小:    {n_unified:,} attr 词")
    if all_dims_in_lib:
        if len(all_dims_in_lib) <= 30:
            print(f"     全部 dim:                   {all_dims_in_lib}")
        else:
            print(f"     前 15 dim:                  {all_dims_in_lib[:15]}")
            print(f"     后 15 dim:                  {all_dims_in_lib[-15:]}")
    if v68_changes:
        print(f"     v6.8 改动说明:")
        for c in v68_changes:
            print(f"       • {c}")

except Exception as e:
    print(f"  ❌ 属性库读取失败: {e}")
PYEOF

    # ── [V6.6 NEW] 自动探测 v6.6 ROI 受限 attr 检索是否启用 ──
    local model_py="${SCRIPT_DIR}/caption_model/model.py"
    local retriever_py="${SCRIPT_DIR}/caption_model/retriever.py"
    if [ -f "$model_py" ] && [ -f "$retriever_py" ]; then
        $PYTHON - "$model_py" "$retriever_py" << 'PYEOF' 2>/dev/null || true
import sys, re
model_py, retriever_py = sys.argv[1], sys.argv[2]

with open(model_py) as f: src = f.read()
with open(retriever_py) as f: rsrc = f.read()

# 探测 K_name / K_attr
k_names = set(re.findall(r'top_k_names=(\d+)', src))
k_attrs = set(re.findall(r'top_k_attrs=(\d+)', src))

# [V6.6 NEW] 探测 ROI 受限 attr 检索是否在 model.py 里启用
has_v66_attr   = 'retrieve_attrs_per_roi_constrained' in src
has_v66_helper = '_category_ids_to_obj_names' in src
has_v66_param  = re.search(r'retrieval_vecs=retrieval_vecs', src) is not None
has_v66_marker = '[V6.6' in src or 'V6.6 核心改动' in src

# [V6.9.3 NEW] 探测 retriever.py 是否有新方法 + 同义词表
has_v693_method = 'def retrieve_attrs_per_roi_constrained' in rsrc
has_v693_alias  = 'COCO_TO_ATTR_ALIAS' in rsrc
has_v693_resolve = 'def _resolve_obj_alias' in rsrc

# 数同义词条数
m_alias = re.search(r"COCO_TO_ATTR_ALIAS\s*:\s*Dict\[str,\s*str\]\s*=\s*\{([^}]+)\}", rsrc)
n_alias = 0
if m_alias:
    n_alias = sum(1 for ln in m_alias.group(1).split('\n')
                  if ':' in ln and not ln.strip().startswith('#'))

print(f"")
print(f"     [v6.6 ROI 受限检索 — 自动探测]")
if k_names:
    print(f"       K_name (LLM 收到的 name token 数): {','.join(sorted(k_names))}")
if k_attrs:
    print(f"       K_attr (LLM 收到的 attr token 数): {','.join(sorted(k_attrs))}")
if k_names and k_attrs:
    try:
        kn = int(list(k_names)[0])
        ka = int(list(k_attrs)[0])
        print(f"       LLM prompt: <vis_32><name_{kn}><attr_{ka}> = {32+kn+ka} 个 visual token")
        print(f"       (v6.5 起默认无文字 marker, _legacy_prompt_v62=True)")
    except Exception: pass

print(f"")
print(f"     [v6.6 model.py 自检]")
if has_v66_attr and has_v66_helper and has_v66_param and has_v66_marker:
    print(f"       ✅ V6.6 ROI 受限 attr 检索 已启用")
    print(f"          • _category_ids_to_obj_names helper")
    print(f"          • retrieve_attrs_per_roi_constrained 调用")
    print(f"          • retrieval_vecs 透传到 _collect_name_attr_tokens_evcap")
else:
    print(f"       ⚠️  V6.6 ROI 受限 attr 检索 未启用 (可能还是旧 v6.5 model.py)")
    if not has_v66_attr:    print(f"          缺: retrieve_attrs_per_roi_constrained 调用")
    if not has_v66_helper:  print(f"          缺: _category_ids_to_obj_names helper")
    if not has_v66_param:   print(f"          缺: retrieval_vecs 参数透传")

print(f"")
print(f"     [v6.9.3 retriever.py 自检]")
if has_v693_method and has_v693_alias and has_v693_resolve:
    print(f"       ✅ V6.9.3 ROI 受限 + 同义词 已就绪")
    print(f"          • retrieve_attrs_per_roi_constrained 方法")
    print(f"          • _resolve_obj_alias 解析方法")
    print(f"          • COCO_TO_ATTR_ALIAS: ~{n_alias} 条同义词")
else:
    print(f"       ⚠️  V6.9.3 retriever 未就绪")
    if not has_v693_method:  print(f"          缺: retrieve_attrs_per_roi_constrained 方法")
    if not has_v693_alias:   print(f"          缺: COCO_TO_ATTR_ALIAS 表")
    if not has_v693_resolve: print(f"          缺: _resolve_obj_alias 方法")
PYEOF
    fi
}

# ════════════════════════════════════════════════════════════════════
# [V6.7 NEW 补丁 2/3] print_gt_stats — 自动识别 Karpathy / COCO 两种格式
# ════════════════════════════════════════════════════════════════════
print_gt_stats() {
    local caption_file="$1"
    local num_caps="${2:-5}"
    if [ -z "$caption_file" ] || [ ! -f "$caption_file" ]; then
        warn "无法统计 GT 文本: 文件不存在 ${caption_file}"
        return
    fi
    echo ""
    echo -e "${CYAN}── GT 文本统计 ──────────────────────────────────────────${NC}"
    $PYTHON - "$caption_file" "$num_caps" << 'PYEOF'
import sys, json
from collections import Counter
caption_file = sys.argv[1]
num_caps = int(sys.argv[2])

with open(caption_file) as f:
    data = json.load(f)

# ★ [V6.7] 自动识别格式 ─────────────────────────────────────────────
if isinstance(data, list):
    # Karpathy 扁平: [{caption, image, image_id}, ...]
    fmt_label = "Karpathy 扁平 (list)"
    total_captions = len(data)
    caps_per_img = Counter()
    images_seen = set()
    for entry in data:
        if not isinstance(entry, dict): continue
        raw_id = entry.get('image_id', entry.get('image', ''))
        iid = str(raw_id).strip()
        if iid.startswith('coco_'): iid = iid[5:]
        caps_per_img[iid] += 1
        img_path = entry.get('image', '')
        if img_path: images_seen.add(img_path)
    n_imgs_with_caps = len(caps_per_img)
    total_images = len(images_seen) if images_seen else n_imgs_with_caps
elif isinstance(data, dict) and 'annotations' in data:
    # COCO 原版: {images: [...], annotations: [{image_id, caption}, ...]}
    fmt_label = "COCO 原版 (dict with annotations)"
    total_captions = len(data.get('annotations', []))
    total_images = len(data.get('images', []))
    caps_per_img = Counter()
    for ann in data.get('annotations', []):
        caps_per_img[str(ann['image_id'])] += 1
    n_imgs_with_caps = len(caps_per_img)
else:
    print(f"  ⚠️ 未知格式: type={type(data).__name__}  keys={list(data.keys()) if isinstance(data,dict) else 'N/A'}")
    sys.exit(0)

avg_caps = total_captions / max(n_imgs_with_caps, 1)
min_caps = min(caps_per_img.values()) if caps_per_img else 0
max_caps = max(caps_per_img.values()) if caps_per_img else 0
expanded_samples = sum(min(cnt, num_caps) for cnt in caps_per_img.values())

print(f"  caption 文件:    {caption_file}")
print(f"  格式:            {fmt_label}")
print(f"  图像总数:        {total_images}")
print(f"  有 caption 的图: {n_imgs_with_caps}")
print(f"  caption 总数:    {total_captions}")
print(f"  平均每图 caption: {avg_caps:.1f}  (min={min_caps}, max={max_caps})")
print(f"  ── [5GT-EXPAND] 展开统计 ──")
print(f"  num_captions_per_image: {num_caps}")
print(f"  展开后样本总数:  {expanded_samples}")
print(f"  展开倍数:        {expanded_samples / max(n_imgs_with_caps, 1):.1f}×")
PYEOF
    echo -e "${CYAN}─────────────────────────────────────────────────────────${NC}"
}

print_all_paths() {
    echo ""
    echo -e "${CYAN}── v6.7 关键路径清单 ────────────────────────────────────${NC}"
    echo ""
    echo "  ┌─ caption_model 包 (v6.4 拆分版, v6.6 改 model.py + retriever.py) ──"
    echo "  │   包目录:      ${SCRIPT_DIR}/caption_model/"
    echo "  │   ★ model.py:     v6.6.1 (ROI 受限 attr 检索)"
    echo "  │   ★ retriever.py: v6.9.3 (retrieve_attrs_per_roi_constrained 方法)"
    echo "  │   其他文件:    v6.5 (无变化)"
    echo "  └─"
    echo ""
    echo "  ┌─ 视觉编码器 (LAVIS 官方) ────────────────────────────"
    echo "  │   编码器权重:  ${VIS_ENCODER}"
    echo "  │   架构标识:    ${VIS_ARCH}"
    echo "  │   输入分辨率:  ${CLIP_IMG_SIZE}×${CLIP_IMG_SIZE} BICUBIC + CLIP mean/std"
    echo "  └─"
    echo ""
    echo "  ┌─ 模型权重 ────────────────────────────────────────────"
    echo "  │   LLM (Vicuna): ${LLM_PATH}"
    echo "  │   Q-Former 预训练权重: ${QFORMER_WEIGHT:-未找到}"
    echo "  └─"
    echo ""
    echo "  ┌─ 数据集 (COCO 2014 + Karpathy 118k ★ V6.7) ──────────"
    echo "  │   图片 root:           ${COCO_IMAGES}"
    echo "  │   ▸ V6.5 改成父目录, 含 train2014/ + val2014/"
    echo "  │   COCO 原版 caption:   ${COCO_CAPTIONS}"
    echo "  │   ★ Karpathy train:    ${KARPATHY_TRAIN_JSON:-未配置}"
    echo "  │   ▸ V6.7 由 run.sh 通过 --train_json 传给 train.py"
    echo "  └─"
    echo ""
    echo "  ┌─ Instances JSON (★ V6.7 双合并) ──────────────────────"
    echo "  │   train instances: ${COCO_INSTANCES}"
    echo "  │   val   instances: ${COCO_INSTANCES_VAL:-未配置}"
    echo "  │   ▸ Karpathy train.json 同时含 train2014/ + val2014/ 路径,"
    echo "  │     CocoBoxLoader 合并两个 instances 进同一个 _image_anns"
    echo "  └─"
    echo ""
    echo "  ┌─ [V6.8] 检索记忆库 (768 维同源空间, 三源融合) ───────"
    echo "  │   名称库 (name_pkl): ${NAME_PKL}"
    echo "  │   属性库 (attr_pkl): ${ATTR_PKL}"
    echo "  │   桥接索引 (json):   ${ATTR_MAP}"
    echo "  │   ▸ v6.8: VAW + VG attribute_types 合并 → unified attr→dim 表"
    echo "  │   ▸ v6.6 没改 attr_memory.pkl 内容, 只改了检索方式"
    echo "  └─"
    echo ""
    echo "  ┌─ [V6.8 NEW] VG attribute_types 离线分类文件 ─────────"
    echo "  │   依赖文件:    ${VG_ATTR_TYPES_JSON}"
    echo "  │   生成脚本:    ${CLASSIFY_VG_SCRIPT}"
    echo "  │   ▸ build_attr 之前必须先生成此文件"
    echo "  └─"
    echo ""
    echo "  ┌─ 输出目录 ────────────────────────────────────────────"
    echo "  │   output_dir:     ${OUTPUT_DIR}"
    echo "  │   checkpoint_dir: ${CKPT_DIR:-${OUTPUT_DIR}/checkpoints}"
    echo "  │   log_dir:        ${LOG_DIR:-${OUTPUT_DIR}/logs}"
    echo "  └─"
    echo ""
    echo -e "${CYAN}─────────────────────────────────────────────────────────${NC}"
}

# ════════════════════════════════════════════════════════════════════
# [V6.6] 检索路线说明 — 反映 ROI 受限 attr 检索
# ════════════════════════════════════════════════════════════════════
print_retrieval_route() {
    echo ""
    echo -e "${CYAN}── v6.6 检索 + 双 Q-Former 路线 (ROI 受限 attr 检索) ────${NC}"
    echo ""
    echo "  阶段 A — vision_frontend (extract_online):"
    echo "    image (PIL) + lvis_boxes (COCO GT bbox)"
    echo "         ↓ LAVIS EVA-ViT-g/14 (Resize 224×224 BICUBIC + CLIP mean/std)"
    echo "    全局: eva_global    [B, 1408]    (整图 mean pool)"
    echo "    全局: eva_patches   [B, 257, 1408]  (CLS + 256 patches)"
    echo "    局部: eva_rois      [B, N, 1408]   (N 个 ROI 特征)"
    echo "    局部: category_ids   [[c1,c2,..]..]  ← v6.6 关键! attr 检索用"
    echo ""
    echo "  阶段 B — model.forward 算 retrieval_qformer (双输出):"
    echo "    eva_patches [B, 257, 1408]"
    echo "         ↓ ln_vision → LAVIS Q-Former (frozen 32q)"
    echo "    全图查询: retrieval_compressed [B, 32, 768]  ← name 检索用"
    echo "    ROI 查询: retrieval_vecs       [B, N, 768]   ← attr 检索用 ★"
    echo ""
    echo "  阶段 C — _collect_name_attr_tokens_evcap (v6.6 核心改动):"
    echo ""
    echo "    [Step 1] name 检索 (v6.5 不变, per-query-token):"
    echo "       global_32q [32,768] → name_keys 全库 → top-8 obj_name"
    echo "       → 填 init_q 前 8 槽 (name_embeds)"
    echo ""
    echo "    [Step 2] ★★★ V6.6 改动: attr 改 ROI 受限检索 ★★★"
    echo "       for ROI in retrieval_vecs[b]:"
    echo "         obj_name = COCO_ID_TO_NAME[category_ids[b][i]]"
    echo "         resolved = _resolve_obj_alias(obj_name)  # couch→sofa"
    echo "         attr_subset = name2attr_idx[resolved]    # 只在该 obj 子集"
    echo "         attr = retrieval_vecs[i] vs attr_subset (top-1, sim≥0.50)"
    echo "       → 填 init_q 后 8 槽 (attr_embeds)"
    echo "       → 实测 sim 0.514 → 0.759 (+0.245), 配对错位修对"
    echo ""
    echo "    [Step 3] 拼接 + 精炼 (v6.5 不变):"
    echo "       init_q [B, 16, 768] = [name_8 | attr_8]"
    echo "       → evcap_qformer.forward_with_queries → refined [B, 16, 768]"
    echo "       → name_tokens [B,8,768] + attr_tokens [B,8,768] (同一语义空间)"
    echo "       ↓"
    echo "  TriStreamFusion(global_32q, vis_32q, name_tokens) → vis_tokens"
    echo "       ↓"
    echo "  阶段 D — LLM prompt:"
    echo "       <BOS><SYS><vis_32><name_8><attr_8><INSTRUCT>  (无文字 marker)"
    echo "       ↓ Vicuna-7B → caption"
    echo ""
    echo "  ── [V6.6 关键诊断数据] (来自 diagnose_roi_attr_match.py 100 张图) ──"
    echo ""
    echo "    方案 A (旧 v6.5 global): mean sim 0.514"
    echo "    方案 B (ROI + 全库):     mean sim 0.849 (+0.335)  但有配对错位"
    echo "    方案 C (本版本受限):     mean sim 0.759 (+0.245)  配对正确 ✓"
    echo ""
    echo "    案例 (motorcycle 图):"
    echo "      方案 A: ROI=person → 'motorcycle:big' (错: motorcycle 属性配给 person)"
    echo "      方案 C: ROI=person → 'person:standing' ✓"
    echo "             ROI=motorcycle → 'motorcycle:red' ✓"
    echo ""
    echo "  ── [V6.9.3] COCO→attr_memory 同义词映射 (修 D1 覆盖 78.8%→~88%) ──"
    echo ""
    echo "    couch→sofa, donut→doughnut, tv→television, cell phone→cellphone"
    echo "    hot dog→hotdog, fire hydrant→fireplug, remote→remote control,"
    echo "    tie→necktie, wine glass→wineglass, sports ball→ball,"
    echo "    dining table→table, hair drier→hair dryer (共 13 条)"
    echo "    救回 8/17 个 COCO 80 类里 attr_memory 覆盖不到的物体"
    echo ""
    echo "  ── [V6.6 推理时降级] ────────────────────────────────────"
    echo ""
    echo "    训练时: vision_frontend 用 GT bbox → ROI 路径完整"
    echo "    推理时 (CocoValBoxLoader 提供 GT): ROI 路径相同 → 也走受限检索"
    echo "    极端情况 (推理无 ROI / 全部 ROI 都不命中):"
    echo "      自动 fallback 走 v6.5 global+per-name 路径 (兼容性保障)"
    echo ""
    echo "  ── [V6.8] attr_memory.pkl (v6.6 完全沿用, 不重建) ──"
    echo ""
    echo "    构建流程: VAW 42 dim + VG 独有 ~52 dim = ~94 dim"
    echo "    依赖文件: vg_attribute_types.json (classify_vg_attrs_v2.py 生成)"
    echo "    PKL retrieve 路径: 完全不变 (v6.6 只是新加方法, 没改旧的)"
    echo ""
    echo -e "${CYAN}─────────────────────────────────────────────────────────${NC}"
}

show_config() {
    echo "配置 v6.7 (${CONFIG_FILE}):"
    echo "  模式:          ${TRAIN_MODE}"
    echo ""
    echo "  ── 视觉编码器 (LAVIS 官方) ───────────────────────────"
    echo "  编码器:        ${VIS_ARCH}"
    echo "  权重路径:      ${VIS_ENCODER:-未配置}"
    echo "  输出维度:      ${CLIP_DIM}d, ${CLIP_PATCHES} patches"
    echo "  输入分辨率:    ${CLIP_IMG_SIZE}×${CLIP_IMG_SIZE} BICUBIC + CLIP mean/std"
    echo ""
    echo "  ── [V6.6] attr 检索: ROI 受限 (改 model.py + retriever.py) ──"
    echo "  attr 检索 query: per-ROI 768d (retrieval_vecs)"
    echo "  attr 检索范围:   该 ROI 物体的 attr 子集 (绕过库内塌缩)"
    echo "  实测 sim:        0.514 → 0.759 (+0.245)"
    echo ""
    echo "  ── [V6.8] attr_memory.pkl (v6.6 沿用不变) ──"
    echo "  unified attr→dim 表: VAW 42 dim + VG 独有 ~52 dim = ~94 dim"
    echo "  实际数 / 三元组数 / LVIS 覆盖率 → 跑 'check' 子命令查看"
    echo ""
    echo "  ── ★ V6.7 数据集 ──────────────────────────────────────"
    echo "  图片 root:     ${COCO_IMAGES}"
    echo "  COCO 原版 cap: ${COCO_CAPTIONS}"
    echo "  Karpathy train: ${KARPATHY_TRAIN_JSON:-未配置}  ★ 由 run.sh 通过 --train_json 传"
    echo "  train inst:    ${COCO_INSTANCES}"
    echo "  val inst:      ${COCO_INSTANCES_VAL:-未配置}    ★ V6.7 双 instances 合并"
    echo ""
    echo "  ── 检索记忆库 (768 维同源空间) ────────────────────────"
    echo "  名称库:        ${NAME_PKL:-未配置}"
    echo "  属性库:        ${ATTR_PKL:-未配置}"
    echo "  桥接索引:      ${ATTR_MAP:-未配置}"
    echo "  VG dim 表:     ${VG_ATTR_TYPES_JSON}"
    echo ""
    echo "  ── 训练设置 ──────────────────────────────────────────"
    echo "  Stage2:        ${STAGE2_EPOCHS}轮 NTP | LS=0.00 | 5GT展开×${NUM_CAPTIONS}"
    echo "  batch:         ${BATCH_SIZE}×${GRAD_ACCUM}=$((BATCH_SIZE * GRAD_ACCUM))"
    echo "  LLM:           ${LLM_PATH}"
    echo "  Q-Former 预训权重: ${QFORMER_WEIGHT:-未找到}"
    echo "  输出:          ${OUTPUT_DIR}"
    echo "  ckpt_dir:      ${CKPT_DIR}"
    echo "  log_dir:       ${LOG_DIR}"
}

# ────────────────────────────────────────────────────────────────────
# [V6.5+-VERIFY] 验证三方对齐 (v6.6 不变, attr_memory.pkl schema 兼容)
# ────────────────────────────────────────────────────────────────────
verify() {
    header "v6.7 三方对齐验证 (attr_memory ↔ name_pkl ↔ caption_model)"
    [ ! -f "$VERIFY_SCRIPT" ] && err "verify_alignment.py 不存在: $VERIFY_SCRIPT"
    [ ! -f "$ATTR_PKL" ] && err "attr_memory.pkl 不存在 (先跑 build_attr): $ATTR_PKL"
    [ ! -f "$NAME_PKL" ] && err "ext_memory_lvis.pkl 不存在: $NAME_PKL"

    step "$PYTHON $VERIFY_SCRIPT"
    cd "$SCRIPT_DIR"
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $PYTHON $VERIFY_SCRIPT
    ok "verify 完成"
    echo ""
    echo "─── 判定标准 ───"
    echo "  [A] attr_memory ↔ caption_model query, 期望 max_sim > 0.3 (理想 >0.5)"
    echo "  [B] name_pkl    ↔ caption_model query, 期望 max_sim > 0.3"
    echo "  [C] 两个 PKL 一致性, 期望差异 < 0.1"
    echo "  ✅ 全部通过 → bash run.sh train_online full"
    echo "  ❌ 有问题 → 检查 attr_memory.pkl 是否用 caption_model 同源 Q-Former"
}

# ────────────────────────────────────────────────────────────────────
# [V6.8-BUILD] 重建 attr_memory.pkl (v6.6 不变)
# ────────────────────────────────────────────────────────────────────
build_attr() {
    header "v6.8 三源 attr_memory 构建 (PACO + VAW + VG via attribute_types)"
    [ ! -f "$BUILD_ATTR_SCRIPT" ] && err "build_attr_memory.py 不存在: $BUILD_ATTR_SCRIPT"

    if [ ! -f "$VG_ATTR_TYPES_JSON" ]; then
        echo -e "${RED}❌ v6.8 必需文件不存在: ${VG_ATTR_TYPES_JSON}${NC}"
        echo ""
        echo "  v6.8 build_attr_memory.py 依赖此文件。请先运行:"
        echo ""
        if [ -f "$CLASSIFY_VG_SCRIPT" ]; then
            echo "    $PYTHON $CLASSIFY_VG_SCRIPT"
        else
            echo "    $PYTHON classify_vg_attrs_v2.py"
            echo ""
            echo "  (脚本路径: $CLASSIFY_VG_SCRIPT, 当前不存在)"
        fi
        echo ""
        echo "  生成的文件格式跟 VAW attribute_types.json 一致:"
        echo "    {dim_name: [attr_words]}, attr 词已跟 VAW 去重 (VAW 优先)"
        exit 1
    else
        ok "v6.8 依赖文件 OK: $VG_ATTR_TYPES_JSON"
        $PYTHON - "$VG_ATTR_TYPES_JSON" << 'PYEOF' 2>/dev/null || true
import sys, json
with open(sys.argv[1]) as f: d = json.load(f)
print(f"     [VG types] {len(d)} 个 dim, {sum(len(v) for v in d.values())} 个 attr 词")
PYEOF
    fi

    if [ -f "$ATTR_PKL" ]; then
        warn "已存在 attr_memory.pkl: $ATTR_PKL"
        warn "重建会覆盖现有文件 (将备份为 attr_memory.bak.pkl)"
        local bak="$(dirname "$ATTR_PKL")/attr_memory.bak.pkl"
        cp "$ATTR_PKL" "$bak" 2>/dev/null && ok "已备份: $bak"
    fi

    step "$PYTHON $BUILD_ATTR_SCRIPT"
    cd "$SCRIPT_DIR"
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $PYTHON $BUILD_ATTR_SCRIPT
    ok "构建完成"
    echo ""
    echo "下一步:  bash run.sh verify"
}

check() {
    header "环境检查 (V6.7 Karpathy 118k + V6.6 ROI 受限 attr 检索 + V6.8 attr_memory)"
    show_config; echo ""

    print_all_paths
    print_retrieval_route

    echo "关键文件检查:"
    _ck() { [ -e "$2" ] && echo -e "  ${GREEN}✓${NC} [$1] $2" \
                        || echo -e "  ${YELLOW}!${NC} [$1] $2 (不存在)"; }
    _ck "视觉编码器 (LAVIS)"        "$VIS_ENCODER"
    _ck "LLM (Vicuna-7B)"           "$LLM_PATH"
    _ck "Q-Former 预训练权重"        "$QFORMER_WEIGHT"
    _ck "COCO 图片 root"             "$COCO_IMAGES"
    _ck "COCO 原版 Caption JSON"     "$COCO_CAPTIONS"
    _ck "★ Karpathy train JSON"     "$KARPATHY_TRAIN_JSON"
    _ck "COCO train Instances JSON"  "$COCO_INSTANCES"
    _ck "★ COCO val Instances JSON" "$COCO_INSTANCES_VAL"
    _ck "lavis_eva_vit.py"           "${SCRIPT_DIR}/lavis_eva_vit.py"
    _ck "caption_model 包目录"        "${SCRIPT_DIR}/caption_model"
    _ck "caption_model/__init__.py"   "${SCRIPT_DIR}/caption_model/__init__.py"
    _ck "caption_model/model.py [V6.6]"       "${SCRIPT_DIR}/caption_model/model.py"
    _ck "caption_model/retriever.py [V6.9.3]" "${SCRIPT_DIR}/caption_model/retriever.py"
    _ck "caption_model/retrieval_qformer.py"  "${SCRIPT_DIR}/caption_model/retrieval_qformer.py"
    _ck "train.py [V6.1]"             "${SCRIPT_DIR}/train.py"
    _ck "config.yaml"                 "${SCRIPT_DIR}/config.yaml"
    _ck "config_loader.py [V6.5]"     "${SCRIPT_DIR}/config_loader.py"
    _ck "config.py (wrapper)"         "${SCRIPT_DIR}/config.py"
    _ck "verify_alignment.py [V6.5+]" "${SCRIPT_DIR}/verify_alignment.py"
    _ck "build_attr_memory.py [V6.8]" "${SCRIPT_DIR}/build_attr_memory.py"
    _ck "classify_vg_attrs_v2.py [V6.8]" "${CLASSIFY_VG_SCRIPT}"
    _ck "vg_attribute_types.json [V6.8]" "${VG_ATTR_TYPES_JSON}"

    # ★ V6.7: 统计的是 Karpathy 训练 JSON (如果配了), 否则 COCO 原版
    local _CAP_FOR_STATS="${KARPATHY_TRAIN_JSON:-$COCO_CAPTIONS}"
    print_gt_stats "$_CAP_FOR_STATS" "$NUM_CAPTIONS"

    echo ""
    echo "[V6.8] 检索记忆库详情 (768 维同源空间):"
    _ck "名称库 (name_pkl)"    "$NAME_PKL"
    _ck "属性库 (attr_pkl)"    "$ATTR_PKL"
    _ck "桥接索引 (attr_map)"  "$ATTR_MAP"

    if [ -f "$NAME_PKL" ] && [ -f "$ATTR_PKL" ]; then
        $PYTHON - "$NAME_PKL" << 'PYEOF' 2>/dev/null || true
import sys, pickle, numpy as np
name_pkl = sys.argv[1]
try:
    with open(name_pkl, 'rb') as f: nm = pickle.load(f)
    if isinstance(nm, dict):
        nk = nm.get('keys', nm.get('key'))
        nv = nm.get('values', nm.get('value'))
    elif isinstance(nm, (list, tuple)) and len(nm) == 2:
        nk, nv = nm
    else: nk = None
    if nk is not None and isinstance(nk, np.ndarray):
        print(f"  ✅ 名称库: keys{list(nk.shape)} | {len(nv) if nv is not None else '?'} 类")
except Exception as e:
    print(f"  ❌ 名称库读取失败: {e}")
PYEOF
        # [V6.8] 增强版 attr_memory.pkl 显示 (v6.6 自检也在里面)
        print_attr_memory_v68 "$ATTR_PKL"
        $PYTHON - "$ATTR_MAP" << 'PYEOF' 2>/dev/null || true
import sys, json
attr_map = sys.argv[1]
try:
    with open(attr_map, 'r') as f: jmap = json.load(f)
    print(f"  ✅ 桥接索引: {len(jmap)} 物体")
except Exception as e:
    print(f"  ❌ 桥接索引读取失败: {e}")
PYEOF
    fi

    if [ -f "$VIS_ENCODER" ]; then
        echo ""
        echo "视觉编码器权重检查:"
        $PYTHON - "$VIS_ENCODER" << 'PYEOF' 2>/dev/null || true
import sys, os, torch
path = sys.argv[1]
size_mb = os.path.getsize(path) / (1024*1024)
print(f"  ✅ 路径: {path}")
print(f"  ✅ 大小: {size_mb:.1f} MB  (预期 ~2000 MB)")
try:
    sd = torch.load(path, map_location='cpu', weights_only=False)
    if 'model' in sd: sd = sd['model']
    if 'state_dict' in sd: sd = sd['state_dict']
    keys = list(sd.keys()) if isinstance(sd, dict) else []
    print(f"  ✅ state_dict keys: {len(keys)}")
    has_patch = any('patch_embed' in k for k in keys)
    has_blocks = any('blocks.' in k for k in keys)
    print(f"  ✅ patch_embed: {has_patch}, blocks.*: {has_blocks}")
except Exception as e:
    print(f"  ❌ 权重加载失败: {e}")
PYEOF
    fi

    echo ""
    echo "GPU:"
    $PYTHON -c "
import torch
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print(f'  {p.name}  显存: {p.total_memory/1e9:.1f}GB')
    ok_str = '✅ 够用' if p.total_memory > 36e9 else '⚠️  可能不够(建议>=40GB)'
    print(f'  v6.7 在线训练 (LAVIS ViT-g + Q-Former): {ok_str}')
else:
    print('  ⚠️ 未检测到 GPU')
" 2>/dev/null || echo "  GPU 检查失败"

    echo ""
    echo "v6.7 架构验证 (caption_model 包导入断言):"
    $PYTHON -c "
import sys
sys.path.insert(0, '${SCRIPT_DIR}')
try:
    from lavis_eva_vit import VisionTransformer, interpolate_pos_embed
    print('  ✅ [lavis_eva_vit] VisionTransformer ✓')

    from caption_model import CaptionModel, _clean_caption
    print('  ✅ [V6.4-PKG]      caption_model 包导入成功')

    # [V6.6] 检查 model.py 是否有 v6.6 的 helper
    if hasattr(CaptionModel, '_category_ids_to_obj_names'):
        print('  ✅ [V6.6-MODEL]    _category_ids_to_obj_names helper 存在')
    else:
        print('  ⚠️  [V6.6-MODEL]    _category_ids_to_obj_names 不存在 (可能还是旧 v6.5 model.py)')

    # [V6.9.3] 检查 retriever.py 是否有新方法
    from caption_model.retriever import PKLMemoryRetriever, COCO_TO_ATTR_ALIAS
    if hasattr(PKLMemoryRetriever, 'retrieve_attrs_per_roi_constrained'):
        print(f'  ✅ [V6.9.3-RETR]   retrieve_attrs_per_roi_constrained 方法存在')
        print(f'  ✅ [V6.9.3-ALIAS]  COCO_TO_ATTR_ALIAS: {len(COCO_TO_ATTR_ALIAS)} 条同义词')
    else:
        print('  ⚠️  [V6.9.3-RETR]   retrieve_attrs_per_roi_constrained 不存在')

    from caption_model.retrieval_qformer import LAVISQFormerForRetrieval
    print('  ✅ [V6.4-LAVIS]    LAVISQFormerForRetrieval ✓ (verify_alignment.py 用)')

    from config import get_ablation_config
    cfg = get_ablation_config('full')
    print(f'  ✅ [config v6.5]   experiment.name={cfg.experiment.name}')
    print(f'  ✅ [config v6.5]   coco_train_images={cfg.paths.coco_train_images}')
    print(f'  ✅ [config v6.5]   coco_karpathy_train_json={cfg.paths.coco_karpathy_train_json or \"(空, 由 yaml 或 --train_json 传)\"}')
    print(f'  ✅ [config v6.5]   lvis_val_json={cfg.paths.lvis_val_json}')

except AssertionError as e:
    print(f'  ❌ 断言失败: {e}')
except ImportError as e:
    print(f'  ❌ 导入失败: {e}')
except Exception as e:
    print(f'  ⚠️  验证跳过: {e}')
" 2>/dev/null || echo "  ⚠️ 模型文件检查跳过"

    echo ""
    echo "train.py [V6.1] 验证:"
    $PYTHON -c "
import sys
sys.path.insert(0, '${SCRIPT_DIR}')
try:
    from train import OnlineCaptionDataset, online_collate_fn, Trainer, CocoBoxLoader
    print('  ✅ [train.py v6.1] OnlineCaptionDataset + Trainer + CocoBoxLoader ✓')
    from train import _parse_karpathy_image_id
    assert _parse_karpathy_image_id('coco_546188') == '546188'
    assert _parse_karpathy_image_id('546188') == '546188'
    assert _parse_karpathy_image_id(546188) == '546188'
    print('  ✅ [V6.1-HELPER] _parse_karpathy_image_id 工作正常')
    import inspect
    sig = inspect.signature(CocoBoxLoader.__init__)
    assert 'val_json_path' in sig.parameters
    print('  ✅ [V6.1-DUAL]   CocoBoxLoader.val_json_path 参数已加')
except Exception as e:
    print(f'  ⚠️  train.py 验证跳过: {e}')
" 2>/dev/null || echo "  ⚠️ train.py 检查跳过"
}

train_online() {
    local ABLATION="${1:-full}"
    local RESUME="" LS_EPOCH="" NUM_CAPS_ARG="" LABEL_SMOOTHING=""
    shift 2>/dev/null || true
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --resume)             RESUME="${2:-auto}"; shift 2 ;;
            --ls_curriculum)      LS_EPOCH="$2"; shift 2 ;;
            --num_captions)       NUM_CAPS_ARG="$2"; shift 2 ;;
            --label_smoothing)    LABEL_SMOOTHING="$2"; shift 2 ;;
            *) shift ;;
        esac
    done

    header "在线训练 V6.7 Karpathy 118k + V6.6 ROI 受限 attr 检索 (消融=${ABLATION})"
    show_config

    print_all_paths
    print_retrieval_route

    [ ! -f "$TRAIN_SCRIPT" ] && err "脚本不存在: $TRAIN_SCRIPT"
    [ ! -d "${SCRIPT_DIR}/caption_model" ] && err "caption_model 包不存在"
    [ ! -f "${SCRIPT_DIR}/caption_model/__init__.py" ] && err "caption_model/__init__.py 不存在"

    if [ ! -f "$ATTR_PKL" ]; then
        warn "attr_memory.pkl 不存在: $ATTR_PKL"
        warn "建议先跑: bash run.sh build_attr"
        warn "  (build_attr 之前要求 vg_attribute_types.json 已生成)"
    elif [ ! -f "${SCRIPT_DIR}/.alignment_verified" ]; then
        warn "未验证 attr_memory ↔ caption_model 对齐"
        warn "强烈建议先跑: bash run.sh verify"
    fi

    # ★ V6.7: 优先打印 Karpathy JSON 的统计 (如果配了)
    local _CAP_FOR_STATS="${KARPATHY_TRAIN_JSON:-$COCO_CAPTIONS}"
    local CAPS_FOR_TRAIN="${NUM_CAPS_ARG:-${NUM_CAPTIONS:-5}}"
    print_gt_stats "$_CAP_FOR_STATS" "$CAPS_FOR_TRAIN"

    if [ -z "$VIS_ENCODER" ] || [ ! -f "$VIS_ENCODER" ]; then
        err "视觉编码器不存在: ${VIS_ENCODER:-未配置}"
    else
        ok "视觉编码器: $VIS_ENCODER ($(du -h "$VIS_ENCODER" | cut -f1))"
    fi
    if [ -z "$NAME_PKL" ] || [ ! -f "$NAME_PKL" ]; then
        warn "名称库不存在: ${NAME_PKL:-未配置}"
    else
        ok "名称库: $NAME_PKL"
    fi
    if [ -z "$ATTR_PKL" ] || [ ! -f "$ATTR_PKL" ]; then
        warn "属性库不存在: ${ATTR_PKL:-未配置}"
    else
        ok "属性库: $ATTR_PKL"
        $PYTHON - "$ATTR_PKL" << 'PYEOF' 2>/dev/null || true
import sys, pickle
from collections import Counter
with open(sys.argv[1], 'rb') as f: am = pickle.load(f)
sources = am.get('sources', [])
n_dims = am.get('n_dims', None)
version = am.get('version', '?')
parts = []
if sources:
    sc = Counter(sources)
    parts = [f"{s}={sc.get(s,0)}" for s in ['PACO', 'VAW', 'VG'] if sc.get(s,0)>0]
if parts:
    print(f"  [V6.8] 三源占比: {' + '.join(parts)} = {sum(Counter(sources).values())}")
if n_dims is not None:
    print(f"  [V6.8] 入库 dim 数: {n_dims} (version={version})")
PYEOF
    fi

    CMD="$PYTHON ${TRAIN_SCRIPT} --ablation ${ABLATION}"
    CMD="$CMD --num_captions ${CAPS_FOR_TRAIN}"

    # ★★★ [V6.7 NEW 补丁 3/3] 显式传 --train_json + --lvis_val_json ★★★
    if [ -n "$KARPATHY_TRAIN_JSON" ] && [ -f "$KARPATHY_TRAIN_JSON" ]; then
        CMD="$CMD --train_json ${KARPATHY_TRAIN_JSON}"
        ok "[V6.7] Karpathy train JSON: $KARPATHY_TRAIN_JSON"
    else
        warn "[V6.7] coco_karpathy_train_json 未配置或不存在: ${KARPATHY_TRAIN_JSON:-未配置}"
        warn "        将退回 COCO 原版 caption (config.paths.coco_train_captions)"
    fi
    if [ -n "$COCO_INSTANCES_VAL" ] && [ -f "$COCO_INSTANCES_VAL" ]; then
        CMD="$CMD --lvis_val_json ${COCO_INSTANCES_VAL}"
        ok "[V6.7] val instances JSON: $COCO_INSTANCES_VAL  (CocoBoxLoader 双合并)"
    else
        warn "[V6.7] lvis_val_json 未配置或不存在: ${COCO_INSTANCES_VAL:-未配置}"
        warn "        Karpathy 训练时 val2014/ 那部分图将无 box 标注"
    fi

    if [ -n "$QFORMER_WEIGHT" ] && [ -f "$QFORMER_WEIGHT" ]; then
        CMD="$CMD --blip2_qformer_path ${QFORMER_WEIGHT}"
        ok "Q-Former 预训练权重: $QFORMER_WEIGHT"
    else
        warn "Q-Former 预训练权重未找到, 将使用随机初始化"
    fi

    if [ -n "$LS_EPOCH" ]; then
        CMD="$CMD --ls_curriculum_epoch ${LS_EPOCH}"
        ok "LS Curriculum: ${LS_EPOCH}ep 后 LS=0.1"
    fi

    if [ -n "$LABEL_SMOOTHING" ]; then
        CMD="$CMD --label_smoothing ${LABEL_SMOOTHING}"
        ok "Label Smoothing: ${LABEL_SMOOTHING}"
    fi

    if [ -n "$RESUME" ]; then
        if [ "$RESUME" = "auto" ]; then
            local _CKDIR="${CKPT_DIR:-${OUTPUT_DIR}/checkpoints_v66_roi}"
            local _OLD_CKDIR="${OUTPUT_DIR}/checkpoints"

            # ── [V6.6] 续训自动找逻辑: 优先新目录, 没找到再去旧 v6.5 目录 ──
            if [ -d "$_CKDIR" ]; then
                RESUME=$(ls -t \
                    "${_CKDIR}"/stage2_ep*.pt \
                    "${_CKDIR}"/stage2_final.pt \
                    "${_CKDIR}"/final.pt \
                    2>/dev/null | head -1)
            fi

            # 新目录没找到 → 退到旧 v6.5 ckpt 目录 (作为 v6.6 起点)
            if [ -z "$RESUME" ] && [ -d "$_OLD_CKDIR" ]; then
                RESUME=$(ls -t \
                    "${_OLD_CKDIR}"/stage2_ep*.pt \
                    "${_OLD_CKDIR}"/stage2_final.pt \
                    "${_OLD_CKDIR}"/final.pt \
                    2>/dev/null | head -1)
                if [ -n "$RESUME" ]; then
                    warn "[V6.6] 新目录 ${_CKDIR} 还没 ckpt"
                    warn "       回退到 v6.5 旧目录的 ckpt: $RESUME"
                    warn "       (会作为 v6.6 训练起点, 但写入新目录 ${_CKDIR})"
                fi
            fi

            if [ -n "$RESUME" ]; then
                ok "续训 ckpt: $RESUME"
                # ── [V6.6] 续训重要提示 ──
                warn "[V6.6] 续训重要提示:"
                warn "       旧 ckpt 是用 v6.5 global attr 训的 (sim 0.514)"
                warn "       新代码用 v6.6 ROI 受限检索 (sim 0.759)"
                warn "       attr_tokens 输入分布会变 → 可能 ep1 短暂震荡"
                warn "       判断标准:"
                warn "         1) 看 1 epoch 趋势, 别只看 ep1 绝对数"
                warn "         2) 跟 v6.5 ep1=1.355 对比, 期望 v6.6 ep1 ≥ 1.38"
                warn "         3) 若 ep1 < 1.355, 看 ep2/ep3 是否爬上来"
                warn "       已用 strict=False 加载, 旧字段如 soft_retriever 会自动跳过"
            else
                warn "无 checkpoint, 从头训练"
            fi
        fi
        if [ -n "$RESUME" ] && [ -f "$RESUME" ]; then
            CMD="$CMD --resume $RESUME"
        fi
    fi

    step "$CMD"
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True eval "$CMD"
    ok "训练完成"
}

show_help() {
    header "图像描述系统 v6.7 | Karpathy 118k + V6.6 ROI + LAVIS"
    show_config; echo ""
    print_all_paths
    print_retrieval_route
    echo "用法: ./run.sh <command> [args]"
    echo ""
    echo "─── 命令列表 ─────────────────────────────────────────────"
    echo "  check                                 环境检查 + 路径清单 + v6.7 自检 ⭐"
    echo "  verify                                [V6.5+] 三方对齐验证"
    echo "  build_attr                            [V6.8]  重建 attr_memory.pkl"
    echo "  train_online [full]                   在线训练 (默认消融=full)"
    echo "  train_online full --resume            续训 (优先新目录, 没找到回退 v6.5 ckpt)"
    echo "  train_online full --resume <ckpt>     指定 checkpoint 续训"
    echo "  train_online full --ls_curriculum 3   前 3 epoch LS=0.00, 之后 LS=0.1"
    echo "  train_online full --label_smoothing 0.1"
    echo "  train_online full --num_captions 5    每图 GT 文本数 (默认 5)"
    echo ""
    echo "─── 推荐工作流 (v6.7 干净路径) ───────────────────────────"
    echo "  0. (一次性) 准备 attr_memory.pkl (跟 v6.8 一样, 已有可跳过):"
    echo "     ${PYTHON:-python} classify_vg_attrs_v2.py     (~5 分钟)"
    echo "     bash run.sh build_attr                        (~70 分钟 GPU)"
    echo "     bash run.sh verify                            (~30 秒)"
    echo "  1. 烟雾测试:    bash run.sh check"
    echo "     → 看到 [V6.6-MODEL] + [V6.9.3-RETR] + [V6.1-HELPER] + [V6.1-DUAL] 都打勾"
    echo "  2. 训 3 epoch:  bash run.sh train_online full"
    echo ""
    echo "─── V6.7 核心变化 (相对 v6.6) ─────────────────────────────"
    echo ""
    echo "  ── 数据集切换 Karpathy 118k (load_config + print_gt_stats + train_online) ──"
    echo "  [LOAD-CFG]      load_config() 加 2 个 yaml_get:"
    echo "                    coco_karpathy_train_json → KARPATHY_TRAIN_JSON"
    echo "                    lvis_val_json             → COCO_INSTANCES_VAL"
    echo "  [GT-STATS-FMT]  print_gt_stats() 用 isinstance(data, list) 自动识别格式"
    echo "  [CMD-ARG]       train_online() 命令末尾加:"
    echo "                    --train_json \$KARPATHY_TRAIN_JSON"
    echo "                    --lvis_val_json \$COCO_INSTANCES_VAL"
    echo ""
    echo "  ── 配合 train.py v6.1 (Karpathy 扁平 + 双 instances 合并) ──"
    echo "  [V6.1-HELPER]  _parse_karpathy_image_id: 'coco_546188' → '546188'"
    echo "  [V6.1-DUAL]    CocoBoxLoader.val_json_path: 合并 val instances"
    echo "  [V6.1-AUTO]    OnlineCaptionDataset.__init__ 自动识别格式"
    echo "  [V6.1-ARG]     argparse 加 --train_json (优先级最高)"
    echo ""
    echo "─── V6.6 核心变化 (继承不变) ─────────────────────────────"
    echo ""
    echo "  ── attr 检索改 ROI 受限 (model.py + retriever.py) ──"
    echo "  [V6.6-ATTR]    attr 检索改 ROI 受限 (sim 0.514 → 0.759)"
    echo "  [V6.9.3-API]   retriever.py 加 retrieve_attrs_per_roi_constrained"
    echo "  [V6.6-COMPAT]  完全向后兼容旧 ckpt"
    echo "  [V6.6-CKPT]    ckpt_dir 加 _v66_roi 后缀, 旧 v6.5 ckpt 完整保留"
    echo ""
    echo "─── V6.8 核心变化 (继承不变) ─────────────────────────────"
    echo ""
    echo "  [V6.8-DEP]      新依赖: vg_attribute_types.json"
    echo "  [V6.8-MERGE]    启动时合并 VAW + VG 为 unified attr→dim 表 (~94 dim)"
    echo ""
    echo "─── V6.5+ 三源融合 (继承不变) ─────────────────────────────"
    echo "  [V6.5+-ATTR]    PACO + VAW + VG 三源融合"
    echo "  [V6.5+-VERIFY]  bash run.sh verify"
    echo "  [V6.5+-BUILD]   bash run.sh build_attr"
    echo ""
    echo "─── V6.4 / V6.3 (继承不变) ────────────────────────────────"
    echo "  [V6.4-PKG]      caption_model/ 包结构 (16 文件)"
    echo "  [V6.4-LAVIS]    Q-Former 用 EVCap 原版 lavis_qformer.BertLMHeadModel"
    echo "  [LAVIS-VIS]     视觉编码器: LAVIS eva_vit_g.pth"
    echo "  [PER-Q]         per-query-token 检索"
    echo "  [STAGE2]        stage2_epochs=3"
    echo ""
    echo "─── 必要文件清单 ─────────────────────────────────────────"
    echo "  视觉编码器权重:        $VIS_ENCODER"
    echo "  Q-Former 预训练权重:   $QFORMER_WEIGHT"
    echo "  LLM (Vicuna-7B):       $LLM_PATH"
    echo "  名称检索库:            $NAME_PKL"
    echo "  属性检索库 [V6.8]:     $ATTR_PKL"
    echo "  桥接索引 (JSON):       $ATTR_MAP"
    echo "  COCO 图片 root:        $COCO_IMAGES"
    echo "  COCO 原版 Caption:     $COCO_CAPTIONS"
    echo "  ★ Karpathy train:     $KARPATHY_TRAIN_JSON"
    echo "  ★ COCO val instances: $COCO_INSTANCES_VAL"
    echo ""
    echo "  caption_model/ 包:     ${SCRIPT_DIR}/caption_model/"
    echo "  ★ model.py [V6.6.1]:  ${SCRIPT_DIR}/caption_model/model.py"
    echo "  ★ retriever.py [V6.9.3]: ${SCRIPT_DIR}/caption_model/retriever.py"
    echo "  ★ train.py [V6.1]:    ${SCRIPT_DIR}/train.py"
    echo "  ★ run.sh  [V6.7]:     ${SCRIPT_DIR}/run.sh"
    echo "  ★ config_loader.py [V6.5]: ${SCRIPT_DIR}/config_loader.py"
    echo "  config.yaml:           ${SCRIPT_DIR}/config.yaml"
    echo "  config.py (wrapper):   ${SCRIPT_DIR}/config.py"
    echo "  verify_alignment.py [V6.5+]:    ${SCRIPT_DIR}/verify_alignment.py"
    echo "  build_attr_memory.py [V6.8]:    ${SCRIPT_DIR}/build_attr_memory.py"
    echo "  classify_vg_attrs_v2.py [V6.8]: ${CLASSIFY_VG_SCRIPT}"
    echo "  vg_attribute_types.json [V6.8]: ${VG_ATTR_TYPES_JSON}"
    echo ""
    echo "─── 输出目录布局 ──────────────────────────────────────────"
    echo "  ${OUTPUT_DIR}/"
    echo "    ├── checkpoints/             ← 旧 v6.5 ckpt (ep1=1.355) 保留"
    echo "    ├── checkpoints_v66_roi/     ← 新 v6.6/v6.7 写入这里"
    echo "    ├── logs/                    ← 旧 v6.5 日志保留"
    echo "    └── logs_v66_roi/            ← 新 v6.6/v6.7 日志写入这里"
}

header "图像描述系统 v6.7 | Karpathy 118k + V6.6 ROI + LAVIS"
setup_python
load_config

case "${1:-help}" in
    check)          check ;;
    verify)         verify ;;
    build_attr)     build_attr ;;
    train_online)   shift; train_online "$@" ;;
    help|--help|-h) show_help ;;
    *)              err "未知命令: $1\n./run.sh help" ;;
esac

echo ""
ok "完成!"