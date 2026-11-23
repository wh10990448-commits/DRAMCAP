"""Same 497-video, 20-reference, beam-3 validation protocol as CIDEr 73.96."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from evaluate_video import main as evaluate_predictions
from .data import EventDataset, collate_events
from .model import EventCaptioner
from .train import atomic_json, move


def main(args):
    dataset = EventDataset(args.split_root, args.cache, args.spatial_cache,
                           args.event_cache, 'val', limit=args.limit)
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=0, collate_fn=collate_events)
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    variant = args.variant
    model = EventCaptioner(str(Path(args.root) / 'models/vicuna-7b-v1.5'),
                           Path(args.root) / 'stage2_final.pt', variant).to(args.device).eval()
    for checkpoint in args.checkpoints:
        path = Path(checkpoint)
        tag = path.stem
        metric_path = output / (tag + '_metrics.json')
        if metric_path.exists() and not args.force:
            continue
        state = torch.load(path, map_location='cpu', weights_only=True)
        if state['feature_fingerprint'] != dataset.metadata['fingerprint'] or state[
                'spatial_fingerprint'] != dataset.spatial_metadata['fingerprint']:
            raise ValueError('Baseline/checkpoint feature provenance mismatch')
        if 'event_fingerprint' in state and state['event_fingerprint'] != dataset.event_metadata['fingerprint']:
            raise ValueError('Event cache provenance mismatch')
        if 'relation_events' in state:
            model.load_adapted_state(state)
        else:
            model.load_baseline(state)
            model.events.scale.data.zero_()
        prediction_path = output / (tag + '_predictions.jsonl')
        temporary = prediction_path.with_suffix('.jsonl.tmp')
        with temporary.open('w') as stream:
            for batch in loader:
                with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
                    captions = model.generate(move(batch, args.device), max_new_tokens=40,
                                              min_new_tokens=3, num_beams=3, do_sample=False,
                                              repetition_penalty=1.05, no_repeat_ngram_size=3)
                for identity, caption in zip(batch['video_ids'], captions):
                    if not caption.strip():
                        raise RuntimeError('Empty caption')
                    stream.write(json.dumps({'video_id': identity, 'split': 'val',
                                             'caption': caption.strip()}) + '\n')
        temporary.replace(prediction_path)
        if args.limit:
            print('SMOKE_PREDICTIONS_ONLY', prediction_path, flush=True)
            continue
        evaluate_predictions(argparse.Namespace(
            references=str(Path(args.split_root) / 'val.jsonl'), predictions=str(prediction_path),
            split='val', output=str(metric_path)))
        report = json.loads(metric_path.read_text())
        report.update(checkpoint=str(path), checkpoint_epoch=state.get('epoch'),
                      variant=variant if 'relation_events' in state else 'A',
                      test_set_used_for_selection=False)
        atomic_json(metric_path, report)
        print('EVENT_VALIDATION', report['variant'], tag, report['scores_times_100']['CIDEr'], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default='/root/autodl-tmp/DRAMCap_video')
    parser.add_argument('--split-root', required=True)
    parser.add_argument('--cache', required=True)
    parser.add_argument('--spatial-cache', required=True)
    parser.add_argument('--event-cache', required=True)
    parser.add_argument('--checkpoints', nargs='+', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--variant', choices=['B', 'C', 'D'], default='B')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--force', action='store_true')
    main(parser.parse_args())
