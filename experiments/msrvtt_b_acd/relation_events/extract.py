"""Extract frozen RoI appearance at EXACT existing spatial boxes/timestamps.

This uses the detector recorded in the 73.96 spatial cache (public torchvision
Faster R-CNN MobileNet); it does not silently replace it with YOLO. Associations
use the eight existing keyframes and are explicitly sparse hypotheses. More
dense tracking is a separate future experiment, not hidden extra input here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import torch
from torchvision.transforms.functional import to_tensor
from video_captioning.data import load_split, sample_video
from video_captioning.detector import RoiDetector
from .association import make_events


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            result.update(block)
    return result.hexdigest()


@torch.inference_mode()
def roi_appearance(model, images, spatial, device):
    tensors = [to_tensor(image).to(device) for image in images]
    targets = []
    for frame, image in enumerate(images):
        boxes = spatial['object_geometry'][frame, :, :4].float().to(device).clone()
        boxes *= boxes.new_tensor([image.width, image.height, image.width, image.height])
        # Invalid padded boxes still need positive area to pass RoI pooling.
        boxes[~spatial['object_mask'][frame].bool().to(device)] = boxes.new_tensor([0, 0, 1, 1])
        targets.append({'boxes': boxes})
    image_list, resized = model.transform(tensors, targets)
    features = model.backbone(image_list.tensors)
    if isinstance(features, torch.Tensor):
        features = {'0': features}
    roi = model.roi_heads.box_roi_pool(features, [t['boxes'] for t in resized], image_list.image_sizes)
    roi = model.roi_heads.box_head(roi)
    frames, count = spatial['object_ids'].shape
    if roi.shape != (frames * count, 1024):
        raise ValueError('Unexpected frozen RoI dimension: ' + str(roi.shape))
    return roi.reshape(frames, count, 1024).float().cpu()


def main(args):
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    spatial_root = Path(args.spatial_cache)
    spatial_meta = json.loads((spatial_root / 'cache_metadata.json').read_text())
    visual_meta = json.loads((Path(args.cache) / 'cache_metadata.json').read_text())
    if spatial_meta['source_feature_fingerprint'] != visual_meta['fingerprint']:
        raise ValueError('Visual/spatial provenance mismatch')
    if spatial_meta['detector'] != 'torchvision-fasterrcnn-mobilenet-v3-large-320-fpn-coco':
        raise ValueError('This extractor must match the detector used by the baseline spatial cache')
    schema = {'schema': 'relation-events-v1', 'frames': 8, 'maximum_events': 504,
              'feature_fingerprint': visual_meta['fingerprint'],
              'spatial_fingerprint': spatial_meta['fingerprint'],
              'appearance': 'frozen_frcnn_mobilenet_box_head_1024',
              'association': 'class_gated_hungarian_0.75_cosine_0.25_iou',
              'geometry': 'current9_previous9_rate9_time5',
              'shot_boundary': 'scene_token_cosine_below_0.4',
              'sparse_association': True, 'ground_truth_captions_used': False,
              'storage': 'valid_events_with_one_masked_slot_for_empty_videos'}
    schema['association_source_sha256'] = sha256(Path(__file__).with_name('association.py'))
    schema['extractor_source_sha256'] = sha256(__file__)
    schema['fingerprint'] = hashlib.sha256(json.dumps(schema, sort_keys=True).encode()).hexdigest()
    meta_path = output / 'cache_metadata.json'
    if meta_path.exists() and json.loads(meta_path.read_text()) != schema:
        raise ValueError('Refusing to overwrite incompatible event cache')
    if not meta_path.exists():
        temporary_meta = output / ('cache_metadata.' + str(os.getpid()) + '.tmp')
        temporary_meta.write_text(json.dumps(schema, indent=2))
        temporary_meta.replace(meta_path)
    detector = RoiDetector(Path(args.root) / 'models', device=args.device)
    for split in args.splits:
        rows = load_split(Path(args.split_root) / (split + '.jsonl'), split)
        if args.limit:
            rows = rows[:args.limit]
        (output / split).mkdir(exist_ok=True)
        for row in rows[args.shard_index::args.num_shards]:
            destination = output / split / (row['video_id'] + '.pt')
            if destination.exists():
                old = torch.load(destination, map_location='cpu', weights_only=True)
                if old['cache_fingerprint'] != schema['fingerprint']:
                    raise ValueError('Stale event sample')
                continue
            spatial = torch.load(spatial_root / split / (row['video_id'] + '.pt'), map_location='cpu', weights_only=True)
            visual = torch.load(Path(args.cache) / split / (row['video_id'] + '.pt'), map_location='cpu', weights_only=True)
            if (sha256(row['video_path']) != spatial['source_video_sha256']
                    or spatial['source_video_sha256'] != visual['source_video_sha256']):
                raise ValueError('Raw video changed since baseline feature extraction')
            images, times, mask = sample_video(row['video_path'], 8)
            if not torch.equal(mask, spatial['frame_mask']) or not torch.allclose(
                    times, spatial['timestamps'].float(), atol=1e-4, rtol=0):
                raise ValueError('Decoded video does not match baseline timestamps')
            appearance = roi_appearance(detector.model, images, spatial, args.device)
            scene = torch.nn.functional.normalize(visual['visual'].float().mean(1), dim=-1)
            shot_starts = torch.zeros(8, dtype=torch.bool); shot_starts[0] = True
            shot_starts[1:] = (scene[1:] * scene[:-1]).sum(-1) < 0.4
            events = make_events(spatial, appearance, shot_starts)
            # Store actual candidates; the batch collator pads masked slots.
            # This avoids reading 504 padded appearance vectors for sparse videos.
            event_count = max(1, int(events['event_mask'].sum()))
            events = {key: value[:event_count].clone() if key.startswith('event_') else value
                      for key, value in events.items()}
            result = {**events, 'video_id': row['video_id'], 'split': split,
                      'cache_fingerprint': schema['fingerprint'], 'timestamps': times,
                      'source_video_sha256': spatial['source_video_sha256'], 'shot_starts': shot_starts}
            temporary = destination.with_suffix('.pt.' + str(os.getpid()) + '.tmp')
            torch.save(result, temporary); temporary.replace(destination)
            print('EVENT_EXTRACTED', split, row['video_id'], int(events['event_mask'].sum()), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default='/root/autodl-tmp/DRAMCap_video')
    parser.add_argument('--split-root', required=True)
    parser.add_argument('--cache', required=True)
    parser.add_argument('--spatial-cache', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--splits', nargs='+', default=['train', 'val'])
    parser.add_argument('--shard-index', type=int, default=0)
    parser.add_argument('--num-shards', type=int, default=1)
    parser.add_argument('--limit', type=int, default=0)
    args = parser.parse_args()
    if not 0 <= args.shard_index < args.num_shards:
        parser.error('Invalid shard')
    main(args)
