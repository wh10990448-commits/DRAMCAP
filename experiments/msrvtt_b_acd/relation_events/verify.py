"""Exact token parity against the original base and D-supervision coverage."""
import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from video_captioning.model import VideoCaptioner as BaselineCaptioner
from .branch import eligible_order_caption
from .data import EventDataset, collate_events
from .model import EventCaptioner
from .train import move


def main(args):
    dataset = EventDataset(args.split_root, args.cache, args.spatial_cache,
                           args.event_cache, 'train', limit=args.train_limit)
    batch = move(next(iter(DataLoader(dataset, batch_size=2, collate_fn=collate_events))), 'cuda:0')
    baseline = torch.load(args.baseline, map_location='cpu', weights_only=True)
    model = EventCaptioner(str(Path(args.root) / 'models/vicuna-7b-v1.5'),
                           Path(args.root) / 'stage2_final.pt', 'B').cuda().eval()
    model.load_baseline(baseline)
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
        original = BaselineCaptioner.tokens(model, batch)
        differences = {}
        for variant in ('B', 'C', 'D'):
            model.events.variant = variant
            enhanced = model.tokens(batch)
            differences[variant] = [float((a.float() - b.float()).abs().max())
                                    for a, b in zip(original, enhanced)]
    if any(d for items in differences.values() for d in items):
        raise RuntimeError('Zero-gated branch changed baseline tokens')
    eligible = 0; videos_with_events = 0; candidate_count = 0
    examples = []
    for row in dataset.rows:
        event = torch.load(Path(args.event_cache) / 'train' / (row['video_id'] + '.pt'),
                           map_location='cpu', weights_only=True)
        mask = event['event_mask'].bool()
        videos_with_events += int(mask.any()); candidate_count += int(mask.sum())
        quality = event['event_reliability'][:, :4].float().prod(-1).sqrt()
        if not bool((mask & (quality > 0.4) & (event['event_change'] > .1)).any()):
            continue
        for caption in row['references']:
            if eligible_order_caption(caption):
                eligible += 1
                if len(examples) < 30:
                    examples.append({'video_id': row['video_id'], 'caption': caption})
    report = {'zero_gate_max_differences': differences,
              'training_videos': len(dataset.rows), 'videos_with_events': videos_with_events,
              'partial_preflight': bool(args.train_limit),
              'total_event_candidates': candidate_count, 'D_eligible_caption_pairs': eligible,
              'D_weak_supervision_examples': examples,
              'D_semantic_negatives_manually_verified': False,
              'test_set_used': False}
    Path(args.output).write_text(json.dumps(report, indent=2))
    print('INITIALIZATION_VERIFIED', json.dumps(report), flush=True)
    if eligible == 0:
        raise RuntimeError('No eligible D supervision; cannot claim D differs from C')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    parser.add_argument('--split-root', required=True)
    parser.add_argument('--cache', required=True)
    parser.add_argument('--spatial-cache', required=True)
    parser.add_argument('--event-cache', required=True)
    parser.add_argument('--baseline', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--train-limit', type=int, default=0)
    main(parser.parse_args())
