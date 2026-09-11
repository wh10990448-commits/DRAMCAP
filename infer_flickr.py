#!/usr/bin/env python3
"""
infer_flickr.py ── Flickr30k 推理脚本 (适配 caption_model v6.6.1 / train v6.1)
                                       [SIGLIP-LEGACY-RENAMED]
=================================================================

★ 默认参数已写死, 直接跑即可:
    python infer_flickr.py

等价于:
    python infer_flickr.py \
        --ablation full \
        --resume   /root/autodl-tmp/wh5319/silip_clip_caption/image_captioning_final/outputs/checkpoints/stage2_ep1.pt \
        --karpathy_json /root/autodl-tmp/wh5319/all_data/flick30/dataset_flickr30k.json \
        --image_dir     /root/autodl-tmp/wh5319/all_data/flickr30k-images \
        --split    test \
        --output   ./flickr30k_test_pred.json \
        --batch_size 8 \
        --eval_metrics

如需覆盖任意参数, 命令行传 flag 即可, 例如换 ckpt:
    python infer_flickr.py --resume /path/to/another.pt
临时跑 100 张 sanity check:
    python infer_flickr.py --max_images 100
跳过 SPICE (Java 没装时):
    python infer_flickr.py --skip_spice --skip_meteor

评测指标 (--eval_metrics, 默认开):
    BLEU-1/2/3/4  (纯 Python)
    METEOR        (需 Java: apt install default-jre)
    ROUGE-L       (纯 Python)
    CIDEr         (纯 Python)
    SPICE         (需 Java + 首次会下 ~90MB stanford-corenlp jar)

[SIGLIP-LEGACY-RENAMED] 改动 (相对旧版):
    本工程只有单一 EVA-CLIP ViT-g 视觉编码器, 并无 SigLIP / 第二个 CLIP.
    本文件 _generate_batch 里调 self.model.generate 时:
      feats['clip_global']          → feats['eva_global']
      feats['siglip_global']        → feats['eva_global_768']
      feats['siglip_rois']          → feats['eva_rois']
      siglip_patch=...              → eva_patches=...
      siglip_mid_patch=...          → eva_mid_patches=...
      siglip_rois_for_retrieval=... → eva_global_per_roi=...
      siglip_patch_projected=...    → eva_global_per_roi_alt=...
      clip_patches=...              → 已合并入 eva_patches (移除)
    其余字节级保持不变.
"""

import os
import re
import json
import glob
import argparse
import logging
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm


# ═══════════════════════════════════════════════════════════════
# ★★★ 默认配置 (用户可直接修改这里, 或用命令行 flag 覆盖) ★★★
# ═══════════════════════════════════════════════════════════════
DEFAULTS = {
    # ─ 模型 / ckpt ─
    'ablation':     'full',
    'resume':       '/root/autodl-tmp/wh5319/silip_clip_caption/image_captioning_final/outputs/checkpoints/stage2_final.pt',
    'blip2_qformer_path': None,
    'device':       'cuda:0',

    # ─ 数据 ─
    'image_dir':    '/root/autodl-tmp/wh5319/all_data/flickr30k-images',
    'karpathy_json':'/root/autodl-tmp/wh5319/all_data/flick30/dataset_flickr30k.json',
    'split':        'test',
    'max_images':   None,          # None = 全部, 或填 100 做 sanity

    # ─ 输出 ─
    'output':       '/root/autodl-tmp/wh5319/silip_clip_caption/image_captioning_final/flickr30k_test_pred.json',
    'save_refs':    False,
    'eval_metrics': True,          # ★ 默认就跑评测
    'skip_spice':   False,
    'skip_meteor':  False,

    # ─ 推理控制 ─
    'batch_size':           8,
    'max_new_tokens':       30,
    'min_new_tokens':       6,
    'num_beams':            5,
    'length_penalty':       1.0,
    'no_repeat_ngram_size': 3,
    'print_every':          20,
}


# ── BF16 autocast (与 train.py 一致) ─────────────────────────────
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


from config import get_ablation_config, create_directories
from caption_model import CaptionModel


# ─────────────────────────────────────────────────────────────
# 生成参数 (跟 train.py _EVAL_GEN_KWARGS 一致)
# ─────────────────────────────────────────────────────────────
_EVAL_GEN_KWARGS = dict(
    max_new_tokens=60,
    min_new_tokens=8,
    num_beams=5,
    length_penalty=1.0,
    repetition_penalty=1.0,
    no_repeat_ngram_size=3,
    early_stopping=True,
)


# ─────────────────────────────────────────────────────────────
# 数据收集
# ─────────────────────────────────────────────────────────────
_IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}


def _img_id_from_filename(fn: str) -> str:
    """1000092795.jpg → '1000092795'"""
    stem = os.path.splitext(os.path.basename(fn))[0]
    return stem


def collect_images_from_dir(image_dir: str, max_images: int = None):
    """直接 glob 图片目录"""
    items = []
    for ext in _IMAGE_EXTS:
        items.extend(glob.glob(os.path.join(image_dir, f"*{ext}")))
        items.extend(glob.glob(os.path.join(image_dir, f"*{ext.upper()}")))
    items = sorted(set(items))
    if max_images is not None:
        items = items[:max_images]
    samples = [{'image_id': _img_id_from_filename(p), 'image_path': p} for p in items]
    return samples


def collect_images_from_karpathy(karpathy_json: str, image_dir: str,
                                 split: str = 'test',
                                 max_images: int = None):
    """
    Karpathy split JSON 格式:
      {
        "dataset": "flickr30k",
        "images": [
          {"filename": "1000092795.jpg", "split": "test", "imgid": 0,
           "sentids": [0,1,2,3,4],
           "sentences": [{"raw": "...", ...}, ...]},
          ...
        ]
      }
    """
    with open(karpathy_json, 'r') as f:
        data = json.load(f)
    images = data.get('images', [])

    samples = []
    refs = {}  # img_id -> [caption, ...]
    for img_info in images:
        if img_info.get('split') != split:
            continue
        fn = img_info.get('filename', '')
        if not fn:
            continue
        path = os.path.join(image_dir, fn)
        if not os.path.exists(path):
            continue
        img_id = _img_id_from_filename(fn)
        samples.append({'image_id': img_id, 'image_path': path})
        sents = []
        for s in img_info.get('sentences', []):
            r = s.get('raw', '').strip()
            if r:
                sents.append(r)
        if sents:
            refs[img_id] = sents

    if max_images is not None:
        samples = samples[:max_images]
        refs = {s['image_id']: refs[s['image_id']]
                for s in samples if s['image_id'] in refs}
    return samples, refs


# ─────────────────────────────────────────────────────────────
# 推理主类
# ─────────────────────────────────────────────────────────────
class FlickrInferencer:
    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
        self.device = torch.device(args.device)

        # 1) 配置 + 模型
        self.config = get_ablation_config(args.ablation)
        try:
            create_directories(self.config)
        except Exception:
            pass

        self.logger.info("=" * 70)
        self.logger.info("Flickr30k 推理 (caption_model v6.6.1)")
        self.logger.info(f"  ablation       = {args.ablation}")
        self.logger.info(f"  resume         = {args.resume}")
        self.logger.info(f"  device         = {args.device}")
        self.logger.info(f"  batch_size     = {args.batch_size}")
        self.logger.info(f"  num_beams      = {args.num_beams}")
        self.logger.info(f"  max_new_tokens = {args.max_new_tokens}")
        self.logger.info("=" * 70)

        self.logger.info("正在构建 CaptionModel ...")
        self.model = CaptionModel(self.config).to(self.device)
        self.model.eval()
        total_m = sum(p.numel() for p in self.model.parameters())
        self.logger.info(f"  总参数: {total_m / 1e6:.2f}M")

        # 2) Tokenizer
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.paths.llm_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = 'right'
        # 重要: set_tokenizer 必须在 load_state_dict 之前 (参考 train.py 顺序)
        self.model.set_tokenizer(self.tokenizer)

        # 3) (可选) 先加载 BLIP-2 Q-Former 预训练权重作为底座
        blip2_path = (args.blip2_qformer_path
                      or getattr(self.config.paths, 'blip2_qformer_path', None))
        if blip2_path and os.path.exists(blip2_path):
            self.logger.info(f"  [INIT] 加载 Q-Former 预训练权重: {blip2_path}")
            try:
                self.model.load_blip2_pretrained(blip2_path)
            except Exception as e:
                self.logger.warning(f"  [INIT] Q-Former 预训练加载失败 (忽略): {e}")
        else:
            self.logger.info(f"  [INIT] 跳过 Q-Former 预训练 (resume ckpt 应已覆盖)")

        # 4) 加载训练 checkpoint
        # [SIGLIP-LEGACY-RENAMED] CaptionModel.load_state_dict() 内部会自动 remap
        # 旧 ckpt key (siglip_aligner.* → vis_enhancer.*, siglip_qformer.* → aux_qformer.*)
        if args.resume and os.path.exists(args.resume):
            self.logger.info(f"  [RESUME] 加载: {args.resume}")
            ckpt = torch.load(args.resume, map_location='cpu', weights_only=False)
            state = ckpt.get('model', ckpt)
            missing, unexpected = self.model.load_state_dict(state, strict=False)
            self.logger.info(f"  [RESUME] missing={len(missing)} unexpected={len(unexpected)}")
            if missing:
                self.logger.info(f"  [RESUME] missing 示例(前5): {missing[:5]}")
            if unexpected:
                self.logger.info(f"  [RESUME] unexpected 示例(前5): {unexpected[:5]}")
            saved_ep = ckpt.get('epoch', '?')
            saved_gs = ckpt.get('global_step', '?')
            self.logger.info(f"  [RESUME] 训练时 epoch={saved_ep} gstep={saved_gs}")
        else:
            self.logger.warning("  [RESUME] ⚠️ 未指定或 ckpt 不存在 - 将用随机初始化推理 (结果无意义)")

        # 5) 加载视觉编码器 (LAVIS EVA-ViT-G)
        self.logger.info("  加载视觉编码器 (LAVIS EVA-ViT-G/14) ...")
        self.model.load_vision_encoders()

        # 6) decoder prompt 设为 stage2_late
        if hasattr(self.model, 'decoder') and hasattr(self.model.decoder, 'set_prompt_stage'):
            self.model.decoder.set_prompt_stage("stage2_late")
            self.logger.info("  [PROMPT] decoder prompt_stage = stage2_late")

        # 7) eval()
        self.model.eval()

        # 检索器状态
        if (self.model._pkl_retriever is not None
                and self.model._pkl_retriever._loaded):
            r = self.model._pkl_retriever
            self.logger.info(f"  [PKL] ✅ name_keys={tuple(r.name_keys.shape)} "
                             f"({len(r._name2idx)}类)")
            if r.attr_keys is not None:
                self.logger.info(f"  [PKL] ✅ attr_keys={tuple(r.attr_keys.shape)}")
        else:
            self.logger.warning("  [PKL] ⚠️ PKL 检索器未就绪 - name/attr token 将退化")

    # ──────────────────────────────────────────────────────────
    @torch.no_grad()
    def _generate_batch(self, images, image_ids):
        """
        images:    list[PIL.Image] (RGB)
        image_ids: list[str]
        return:    list[(image_id, caption)]
        """
        feats = self.model.extract_online(images, lvis_boxes=None)

        use_sp = self.config.ablation.use_spatial
        use_at = self.config.ablation.use_attribute

        gen_kwargs = dict(_EVAL_GEN_KWARGS)
        gen_kwargs['max_new_tokens']        = self.args.max_new_tokens
        gen_kwargs['min_new_tokens']        = self.args.min_new_tokens
        gen_kwargs['num_beams']             = self.args.num_beams
        gen_kwargs['length_penalty']        = self.args.length_penalty
        gen_kwargs['no_repeat_ngram_size']  = self.args.no_repeat_ngram_size

        # [SIGLIP-LEGACY-RENAMED] 位置参数 + kwargs 全部改成 eva_*
        #   feats['clip_global']  → feats['eva_global']           (raw 1408d)
        #   feats['siglip_global']→ feats['eva_global_768']       (proj 768d)
        #   feats['siglip_rois']  → feats['eva_rois']             (ROI Align)
        #   siglip_patch          → eva_patches
        #   siglip_mid_patch      → eva_mid_patches
        #   siglip_rois_for_retrieval → eva_global_per_roi
        #   siglip_patch_projected    → eva_global_per_roi_alt
        #   clip_patches              → 已合并入 eva_patches (移除单独 kwarg)
        with _safe_autocast():
            pred_ids = self.model.generate(
                feats['eva_global'],
                feats['eva_global_768'],
                feats['eva_rois'],
                abs_pos=feats.get('absolute_pos'),
                rel_pos=(feats.get('relative_pos') if use_sp else None),
                attr_feats=(feats.get('attribute_features') if use_at else None),
                tag_sequences=feats.get('tag_sequences'),
                is_core=feats.get('is_core'),
                importance_scores=feats.get('importance_scores'),
                roi_mask=(feats.get('roi_mask') if use_sp else None),
                category_ids=feats.get('category_ids'),
                use_qformer=True,
                use_spatial_anchor=use_sp,
                eva_patches=feats.get('eva_patches'),
                eva_mid_patches=feats.get('eva_mid_patches'),
                tag_confidence=feats.get('tag_confidence'),
                eva_global_per_roi=feats.get('eva_global_per_roi'),
                eva_global_per_roi_alt=feats.get('eva_global_per_roi_alt'),
                boxes_batch=feats.get('boxes_batch'),
                **gen_kwargs,
            )

        results = []
        for img_id, pid in zip(image_ids, pred_ids):
            cap = self.model.decode_and_clean(pid, self.tokenizer)
            if not cap or not cap.strip():
                cap = "an image."
            results.append((img_id, cap))
        return results

    # ──────────────────────────────────────────────────────────
    def run(self, samples):
        results = []
        bsz = self.args.batch_size
        total = len(samples)
        if total == 0:
            self.logger.warning("没有任何样本, 直接退出")
            return []

        self.logger.info(f"开始推理: {total} 张图片  batch_size={bsz}")

        pbar = tqdm(range(0, total, bsz), desc="Infer")
        for start in pbar:
            chunk = samples[start: start + bsz]
            imgs, ids = [], []
            for s in chunk:
                try:
                    img = Image.open(s['image_path']).convert('RGB')
                    imgs.append(img)
                    ids.append(s['image_id'])
                except Exception as e:
                    self.logger.warning(f"读图失败 {s['image_path']}: {e}")
            if not imgs:
                continue

            try:
                preds = self._generate_batch(imgs, ids)
            except Exception as e:
                self.logger.error(f"batch 推理失败 (start={start}): {e}")
                import traceback; traceback.print_exc()
                preds = [(i, "") for i in ids]

            for img_id, cap in preds:
                results.append({'image_id': img_id, 'caption': cap})

            if (start // bsz) % max(1, self.args.print_every) == 0 and preds:
                pbar.write(f"  [{start+1}/{total}] {preds[0][0]} → {preds[0][1]}")

        return results


# ─────────────────────────────────────────────────────────────
# 标准 COCO Caption 5 指标 (BLEU / METEOR / ROUGE-L / CIDEr / SPICE)
# ─────────────────────────────────────────────────────────────
def _try_eval_metrics(predictions, references, logger,
                      skip_spice: bool = False, skip_meteor: bool = False):
    """
    标准 COCO Caption 评测:
        BLEU-1/2/3/4  (纯 Python)
        METEOR        (需 Java)
        ROUGE-L       (纯 Python)
        CIDEr         (纯 Python)
        SPICE         (需 Java + 首次自动下 ~90MB CoreNLP jar, 慢)
    """
    if not references:
        return None

    try:
        from pycocoevalcap.bleu.bleu import Bleu
        from pycocoevalcap.cider.cider import Cider
        from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
    except ImportError:
        logger.info("  [METRIC] pycocoevalcap 未安装, 跳过 (pip install pycocoevalcap)")
        return None

    Meteor = None
    Rouge = None
    Spice = None
    if not skip_meteor:
        try:
            from pycocoevalcap.meteor.meteor import Meteor
        except Exception as e:
            logger.warning(f"  [METRIC] METEOR import 失败 (跳过): {e}")
    try:
        from pycocoevalcap.rouge.rouge import Rouge
    except Exception as e:
        logger.warning(f"  [METRIC] ROUGE-L import 失败 (跳过): {e}")
    if not skip_spice:
        try:
            from pycocoevalcap.spice.spice import Spice
        except Exception as e:
            logger.warning(f"  [METRIC] SPICE import 失败 (跳过): {e}")

    # ── 准备 gts / res ──
    gts_raw, res_raw = {}, {}
    for p in predictions:
        iid = str(p['image_id'])
        if iid not in references:
            continue
        gts_raw[iid] = [{'caption': r} for r in references[iid]]
        res_raw[iid] = [{'caption': p['caption']}]
    if not gts_raw:
        logger.info("  [METRIC] 无 image_id 在参考集合中, 跳过")
        return None

    try:
        tok = PTBTokenizer()
        gts = tok.tokenize(gts_raw)
        res = tok.tokenize(res_raw)
    except Exception as e:
        logger.warning(f"  [METRIC] PTBTokenizer 失败 ({e}), 用 split() 兜底")
        gts = {k: [' '.join(re.findall(r"[A-Za-z']+", v[0]['caption'].lower()))]
               for k, v in gts_raw.items()}
        res = {k: [' '.join(re.findall(r"[A-Za-z']+", v[0]['caption'].lower()))]
               for k, v in res_raw.items()}

    metrics = {}

    # BLEU-1/2/3/4
    try:
        bleu_score, _ = Bleu(4).compute_score(gts, res)
        metrics['BLEU-1'] = bleu_score[0]
        metrics['BLEU-2'] = bleu_score[1]
        metrics['BLEU-3'] = bleu_score[2]
        metrics['BLEU-4'] = bleu_score[3]
    except Exception as e:
        logger.warning(f"  [METRIC] BLEU 计算失败: {e}")

    # METEOR
    if Meteor is not None:
        try:
            logger.info("  [METRIC] 计算 METEOR ...")
            meteor_score, _ = Meteor().compute_score(gts, res)
            metrics['METEOR'] = meteor_score
        except Exception as e:
            logger.warning(f"  [METRIC] METEOR 计算失败 (检查 Java 是否安装): {e}")

    # ROUGE-L
    if Rouge is not None:
        try:
            logger.info("  [METRIC] 计算 ROUGE-L ...")
            rouge_score, _ = Rouge().compute_score(gts, res)
            metrics['ROUGE-L'] = rouge_score
        except Exception as e:
            logger.warning(f"  [METRIC] ROUGE-L 计算失败: {e}")

    # CIDEr
    try:
        logger.info("  [METRIC] 计算 CIDEr ...")
        cider_score, _ = Cider().compute_score(gts, res)
        metrics['CIDEr'] = cider_score
    except Exception as e:
        logger.warning(f"  [METRIC] CIDEr 计算失败: {e}")

    # SPICE (慢, 放最后)
    if Spice is not None:
        try:
            logger.info("  [METRIC] 计算 SPICE 中 (首次需下 ~90MB CoreNLP jar, 较慢)...")
            spice_score, _ = Spice().compute_score(gts, res)
            metrics['SPICE'] = spice_score
        except Exception as e:
            logger.warning(f"  [METRIC] SPICE 计算失败: {e}")
            logger.warning(f"               需 Java (apt install default-jre) "
                           f"+ 首次会自动下载 stanford-corenlp jar")

    if metrics:
        logger.info("=" * 70)
        logger.info("评估结果 (Flickr30k):")
        order = ['BLEU-1', 'BLEU-2', 'BLEU-3', 'BLEU-4',
                 'METEOR', 'ROUGE-L', 'CIDEr', 'SPICE']
        for k in order:
            if k in metrics:
                logger.info(f"  {k:<9s} = {metrics[k]:.4f}")
        for k, v in metrics.items():
            if k not in order:
                logger.info(f"  {k:<9s} = {v:.4f}")
        logger.info("=" * 70)
    return metrics


def _save_metrics(metrics, out_path, logger):
    """把 metrics dict 保存成 json, 方便后续比较"""
    if not metrics:
        return
    metrics_path = out_path.replace('.json', '.metrics.json')
    if metrics_path == out_path:
        metrics_path = out_path + '.metrics.json'
    try:
        with open(metrics_path, 'w', encoding='utf-8') as f:
            json.dump({k: float(v) for k, v in metrics.items()},
                      f, ensure_ascii=False, indent=2)
        logger.info(f"✅ 指标已保存: {metrics_path}")
    except Exception as e:
        logger.warning(f"保存 metrics 失败: {e}")


# ─────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Flickr30k 推理 (caption_model v6.6.1 / train v6.1)")
    # 模型 / 配置
    parser.add_argument('--ablation', default=DEFAULTS['ablation'])
    parser.add_argument('--resume',   default=DEFAULTS['resume'])
    parser.add_argument('--blip2_qformer_path', default=DEFAULTS['blip2_qformer_path'])
    parser.add_argument('--device',   default=DEFAULTS['device'])

    # 数据
    parser.add_argument('--image_dir',     default=DEFAULTS['image_dir'])
    parser.add_argument('--karpathy_json', default=DEFAULTS['karpathy_json'])
    parser.add_argument('--split',         default=DEFAULTS['split'],
                        choices=['train', 'val', 'test', 'restval'])
    parser.add_argument('--max_images', type=int, default=DEFAULTS['max_images'],
                        help='只跑前 N 张 (默认 None=全部)')

    # 输出
    parser.add_argument('--output',     default=DEFAULTS['output'])
    parser.add_argument('--save_refs',  action='store_true', default=DEFAULTS['save_refs'])

    # 评测 (默认开, 用 --no_eval_metrics 关掉)
    parser.add_argument('--eval_metrics',    dest='eval_metrics',
                        action='store_true', default=DEFAULTS['eval_metrics'],
                        help='计算 BLEU/METEOR/ROUGE-L/CIDEr/SPICE (默认: 开)')
    parser.add_argument('--no_eval_metrics', dest='eval_metrics',
                        action='store_false',
                        help='关闭评测')
    parser.add_argument('--skip_spice',  action='store_true', default=DEFAULTS['skip_spice'],
                        help='跳过 SPICE (Java 不可用 / 不想等)')
    parser.add_argument('--skip_meteor', action='store_true', default=DEFAULTS['skip_meteor'],
                        help='跳过 METEOR (Java 不可用)')

    # 推理控制
    parser.add_argument('--batch_size',           type=int,   default=DEFAULTS['batch_size'])
    parser.add_argument('--max_new_tokens',       type=int,   default=DEFAULTS['max_new_tokens'])
    parser.add_argument('--min_new_tokens',       type=int,   default=DEFAULTS['min_new_tokens'])
    parser.add_argument('--num_beams',            type=int,   default=DEFAULTS['num_beams'])
    parser.add_argument('--length_penalty',       type=float, default=DEFAULTS['length_penalty'])
    parser.add_argument('--no_repeat_ngram_size', type=int,   default=DEFAULTS['no_repeat_ngram_size'])
    parser.add_argument('--print_every',          type=int,   default=DEFAULTS['print_every'])

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s')
    logger = logging.getLogger("flickr-infer")

    # 入口横幅
    logger.info("┌" + "─" * 68 + "┐")
    logger.info("│" + "  Flickr30k 推理 (默认参数已写死, 可用 CLI flag 覆盖)".ljust(68) + "│")
    logger.info("│" + f"    resume        = {args.resume}".ljust(68)[:68] + "│")
    logger.info("│" + f"    karpathy_json = {args.karpathy_json}".ljust(68)[:68] + "│")
    logger.info("│" + f"    image_dir     = {args.image_dir}".ljust(68)[:68] + "│")
    logger.info("│" + f"    split         = {args.split}    output = {args.output}".ljust(68)[:68] + "│")
    logger.info("│" + f"    eval_metrics  = {args.eval_metrics}    batch_size = {args.batch_size}".ljust(68)[:68] + "│")
    logger.info("└" + "─" * 68 + "┘")

    # 收集样本
    refs = {}
    if args.karpathy_json and os.path.exists(args.karpathy_json):
        if not os.path.isdir(args.image_dir):
            logger.error(f"image_dir 不是目录: {args.image_dir}")
            return
        samples, refs = collect_images_from_karpathy(
            args.karpathy_json, args.image_dir, split=args.split,
            max_images=args.max_images)
        logger.info(f"  [Karpathy] split={args.split}  匹配到 {len(samples)} 张, "
                    f"带参考句子的 {len(refs)} 张")
    else:
        if args.karpathy_json:
            logger.warning(f"  [Karpathy] 文件不存在, 改走目录扫描: {args.karpathy_json}")
        if not os.path.isdir(args.image_dir):
            logger.error(f"image_dir 不是目录: {args.image_dir}")
            return
        samples = collect_images_from_dir(args.image_dir, max_images=args.max_images)
        logger.info(f"  [Dir]  扫到 {len(samples)} 张图片  ({args.image_dir})")

    if not samples:
        logger.error("没有任何样本可推理, 退出")
        return

    # 推理
    inferencer = FlickrInferencer(args, logger)
    predictions = inferencer.run(samples)

    # 保存
    out_path = args.output
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or '.', exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(predictions, f, ensure_ascii=False, indent=2)
    logger.info(f"✅ 预测已保存: {out_path}  ({len(predictions)} 条)")

    if args.save_refs and refs:
        refs_path = out_path + '.refs.json'
        with open(refs_path, 'w', encoding='utf-8') as f:
            json.dump(refs, f, ensure_ascii=False, indent=2)
        logger.info(f"✅ 参考句已保存: {refs_path}  ({len(refs)} 张图)")

    # 评测
    if args.eval_metrics:
        metrics = _try_eval_metrics(
            predictions, refs, logger,
            skip_spice=args.skip_spice,
            skip_meteor=args.skip_meteor,
        )
        _save_metrics(metrics, out_path, logger)


if __name__ == "__main__":
    main()