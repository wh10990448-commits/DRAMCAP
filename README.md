| `name2attr_topk.json` | Object-to-attribute mapping | `paths.attr_map` |
| Trained captioning checkpoint | Inference | `--checkpoint` or `--resume` |

The repository contains the name and attribute memory files. Prepare the visual encoder, language model, BLIP-2 weights, and trained captioning checkpoint separately. The language-model directory should contain the Hugging Face model configuration, weights, and tokenizer files.

## Configuration

Keep the existing structure of `config.yaml` and update its path fields. The following example shows the main fields to replace; it is not a complete replacement for the configuration file.

```yaml
paths:
  project_dir: /path/to/DRAMCAP

  llm_path: /path/to/models/vicuna-7b-v1.5
  eva_vit_g_model: /path/to/models/eva_vit_g.pth
  eva01_model: /path/to/models/eva_vit_g.pth
  blip2_qformer_path: /path/to/models/blip2_pretrained_flant5xxl.pth

  name_pkl: /path/to/DRAMCAP/ext_memory_lvis.pkl
  attr_pkl: /path/to/DRAMCAP/attr_memory/attr_memory.pkl
  attr_map: /path/to/DRAMCAP/attr_memory/name2attr_topk.json

  coco_train_images: /path/to/data/coco_2014
  coco_val_images: /path/to/data/coco_2014
  coco_train_captions: /path/to/data/coco_karpathy_train.json
  coco_val_captions: /path/to/data/coco_karpathy_val.json
  coco_karpathy_train_json: /path/to/data/coco_karpathy_train.json

  lvis_train_json: /path/to/data/coco_2014/annotations/instances_train2014.json
  lvis_val_json: /path/to/data/coco_2014/annotations/instances_val2014.json

  output_dir: /path/to/DRAMCAP/outputs
  features_dir: /path/to/DRAMCAP/outputs/features
  checkpoint_dir: /path/to/DRAMCAP/outputs/checkpoints
  log_dir: /path/to/DRAMCAP/outputs/logs
```

`lvis_train_json` and `lvis_val_json` are historical field names. In the current COCO configuration, they point to COCO instance annotations.

Training epochs, batch size, gradient accumulation, and learning rate are configured under `training`. Adjust batch sizes to fit the available GPU memory.

## COCO Data Preparation

Prepare COCO 2014 images and Karpathy-split caption annotations. Preserve the image-directory structure:

```text
coco_2014/
├── train2014/
├── val2014/
└── annotations/
    ├── instances_train2014.json
    └── instances_val2014.json
```

The training loader supports flat Karpathy-format caption records:

```json
[
  {
    "caption": "A person riding a bicycle on a street.",
    "image": "train2014/COCO_train2014_000000000001.jpg",
    "image_id": "coco_1"
  }
]
```

This record only illustrates the schema. Image names and captions must match the actual dataset.

When the `image` field includes a `train2014/` or `val2014/` prefix, set the training image root to their shared parent, `coco_2014/`. Karpathy training uses images from both original COCO subsets, so prepare both instance-annotation files.

## Training

After updating the configuration, run:

```bash
python train.py \
  --ablation full \
  --image_dir /path/to/data/coco_2014 \
  --train_json /path/to/data/coco_karpathy_train.json \
  --lvis_json /path/to/data/coco_2014/annotations/instances_train2014.json \
  --lvis_val_json /path/to/data/coco_2014/annotations/instances_val2014.json \
  --blip2_qformer_path /path/to/models/blip2_pretrained_flant5xxl.pth
```

The current entry point uses next-token prediction cross-entropy, defaults to `label_smoothing=0.0`, and fixes the number of captions per image to five.

To continue training from a checkpoint:

```bash
python train.py \
  --ablation full \
  --resume /path/to/checkpoints/stage2_ep1.pt
```

The data, pretrained-model, and memory-bank paths must remain correctly configured when continuing training. Checkpoints such as `stage2_ep*.pt` and `stage2_final.pt` are written to the configured checkpoint directory.

## Single-Image Inference

```bash
python infer.py \
  --mode single \
  --checkpoint /path/to/checkpoints/stage2_ep1.pt \
  --image_path /path/to/example.jpg \
  --yoloe_path /path/to/models/yoloe/model.pt
```

To use grid regions instead of detector proposals:

```bash
python infer.py \
  --mode single \
  --checkpoint /path/to/checkpoints/stage2_ep1.pt \
  --image_path /path/to/example.jpg \
  --no_yoloe
```

Add `--no_retrieval_diag` to disable detailed retrieval diagnostics.

## Directory Inference

```bash
python infer.py \
  --mode batch \
  --checkpoint /path/to/checkpoints/stage2_ep1.pt \
  --image_dir /path/to/images \
  --yoloe_path /path/to/models/yoloe/model.pt \
  --output_file batch_predictions.json \
  --max_images 100
```

`--max_images` limits the number of images processed in the run.

## COCO Evaluation

```bash
python infer.py \
  --mode eval \
  --checkpoint /path/to/checkpoints/stage2_ep1.pt \
  --karpathy_json /path/to/data/coco_karpathy_test.json \
  --image_dir /path/to/data/coco_2014/val2014 \
  --yoloe_path /path/to/models/yoloe/model.pt \
  --output_file coco_predictions.json \
  --max_images 5000
```

Set `--image_dir` to the directory that resolves the images referenced by your test annotations. Caption metrics are calculated through the COCO caption evaluation tools when their dependencies are available.

## Flickr30k Inference and Evaluation

```bash
python infer_flickr.py \
  --resume /path/to/checkpoints/stage2_ep1.pt \
  --karpathy_json /path/to/data/dataset_flickr30k.json \
  --image_dir /path/to/data/flickr30k-images \
  --split test \
  --output flickr30k_predictions.json \
  --batch_size 8
```

The Flickr30k entry point uses `--resume` to specify the captioning checkpoint.

| Argument | Purpose |
| --- | --- |
| `--max_images 100` | Run on a smaller subset |
| `--no_eval_metrics` | Generate captions without calculating metrics |
| `--skip_spice` | Skip SPICE |
| `--skip_meteor` | Skip METEOR |
| `--batch_size` | Set the inference batch size |

## Building the Attribute Memory

Skip this step when using the provided memory banks. To rebuild the attribute memory, prepare the relevant PACO, VAW, and Visual Genome images and annotations, and inspect the available arguments:

```bash
python classify_vg_attrs.py --help
python build_attr_memory_mean.py --help
```

These scripts have independent default data paths. Override them through command-line arguments or update their path definitions when moving to a new environment.

## Implementation Notes

- The supplied configuration and some script defaults contain absolute paths from the development environment. Replace them before running the code.
- The command examples use Bash line continuations. In Windows PowerShell, combine each command into one line or use PowerShell continuation syntax.
- `run.sh` retains references to `build_attr_memory.py`, `classify_vg_attrs_v2.py`, and `verify_alignment.py`. Those exact filenames are absent from this snapshot; update the references or provide the scripts before using the corresponding helper commands.
- `eval_results_v621_yoloe.json` contains saved predictions. Evaluate a matching checkpoint with the intended data split and inference protocol to obtain reproducible metrics.

## Acknowledgments

We acknowledge LAVIS, BLIP-2, EVCap, and the COCO, Flickr30k, LVIS, PACO, VAW, Visual Genome, NoCaps, and WHOOPS datasets and benchmarks. The architecture and qualitative-example images in this README are extracted from the accompanying DRAMCap manuscript.
