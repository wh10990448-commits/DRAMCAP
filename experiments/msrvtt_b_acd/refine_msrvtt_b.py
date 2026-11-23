"""Evaluate small changes to the verified B checkpoint on the validation split."""
import argparse
import json
from pathlib import Path
import time

import torch
from torch.utils.data import DataLoader

from evaluate_video import main as evaluate_predictions
from relation_events.data import EventDataset, collate_events
from relation_events.model import EventCaptioner
from relation_events.train import atomic_json, move


DECODE = {
    'max_new_tokens': 40, 'min_new_tokens': 3, 'num_beams': 3,
    'do_sample': False, 'repetition_penalty': 1.05,
    'no_repeat_ngram_size': 3, 'length_penalty': 1.0,
}
TRIALS = {
    'original': {},
    'event_residual_065': {'event_multiplier': 0.65},
    'event_residual_125': {'event_multiplier': 1.25},
    'beam_5': {'num_beams': 5},
    'length_penalty_070': {'length_penalty': 0.70},
    'length_penalty_125': {'length_penalty': 1.25},
    'average_epochs_6_7_8': {'averaged': True},
}


def load_checkpoint(path, dataset):
    state = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
    for key, expected in (
        ('feature_fingerprint', dataset.metadata['fingerprint']),
        ('spatial_fingerprint', dataset.spatial_metadata['fingerprint']),
        ('event_fingerprint', dataset.event_metadata['fingerprint']),
        ('variant', 'B'),
    ):
        if state[key] != expected:
            raise ValueError(f'{path} differs in {key}')
    return state


def average_event_weights(source, dataset, base_state, output):
    states = [load_checkpoint(source / f'epoch_{epoch:03d}.pt', dataset)
              for epoch in (6, 7, 8)]
    weights = (0.25, 0.5, 0.25)
    averaged = {}
    for key, tensor in base_state['relation_events'].items():
        weighted = torch.zeros_like(tensor, dtype=torch.float32)
        for state, weight in zip(states, weights):
            weighted.add_(state['relation_events'][key].float(), alpha=weight)
        averaged[key] = weighted.to(tensor.dtype)
    state = dict(base_state, relation_events=averaged,
                 stage='B_epoch_weight_average', epoch=None,
                 averaged_epochs=[6, 7, 8], averaging_weights=list(weights))
    temporary = output.with_suffix('.tmp')
    torch.save(state, temporary)
    temporary.replace(output)
    return state


def evaluate(model, dataset, output, tag, decode, config):
    loader = DataLoader(dataset, batch_size=4, num_workers=0, collate_fn=collate_events)
    prediction_path = output / f'{tag}_predictions.jsonl'
    metrics_path = output / f'{tag}_metrics.json'
    temporary = prediction_path.with_suffix('.tmp')
    start = time.monotonic()
    seen = 0
    with temporary.open('w') as stream:
        for cpu_batch in loader:
            with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
                captions = model.generate(move(cpu_batch, 'cuda:0'), **decode)
            for identity, caption in zip(cpu_batch['video_ids'], captions, strict=True):
                payload = {'video_id': identity, 'split': 'val', 'caption': caption.strip()}
                stream.write(json.dumps(payload) + '\n')
            seen += len(captions)
            if seen % 100 == 0 or seen == len(dataset):
                atomic_json(output / 'progress.json', {
                    'stage': 'validation', 'trial': tag, 'videos': seen, 'total': len(dataset)})
    temporary.replace(prediction_path)
    evaluate_predictions(argparse.Namespace(references=str(config['references']),
        predictions=str(prediction_path), split='val', output=str(metrics_path)))
    report = json.loads(metrics_path.read_text())
    report.update(config, trial=tag, decode=decode, seconds=round(time.monotonic()-start, 2))
    atomic_json(metrics_path, report)
    print('B_REFINE_RESULT', tag, report['scores_times_100']['CIDEr'], flush=True)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    root, run, output = args.root, args.run, args.output
    output.mkdir(parents=True, exist_ok=True)
    split_dir = root / 'data/msrvtt/caption_splits'
    dataset = EventDataset(split_dir, root / 'runs/msrvtt_eva_recovered_20261115/features',
        root / 'runs/msrvtt_hierarq_spatial_20261115/spatial_features',
        run / 'event_features', 'val')
    source = run / 'B/training'
    checkpoint = source / 'epoch_007.pt'
    known = json.loads((run / 'B/validation/epoch_007_metrics.json').read_text())
    base_state = load_checkpoint(checkpoint, dataset)
    model = EventCaptioner(str(root / 'models/vicuna-7b-v1.5'),
                           root / 'stage2_final.pt', 'B').to('cuda:0').eval()
    results = {}
    for name, settings in TRIALS.items():
        remaining = dict(settings)
        factor = remaining.pop('event_multiplier', 1.0)
        active_checkpoint = checkpoint
        if remaining.pop('averaged', False):
            active_checkpoint = output / 'average_epochs_6_7_8.pt'
            state = average_event_weights(source, dataset, base_state, active_checkpoint)
        else:
            state = base_state
        model.load_adapted_state(state)
        if factor != 1.0:
            with torch.no_grad():
                model.events.scale.copy_(torch.atanh(model.events.scale.tanh() * factor))
        report = evaluate(model, dataset, output, name, {**DECODE, **remaining}, {
            'references': str(split_dir / 'val.jsonl'), 'checkpoint': str(active_checkpoint),
            'event_multiplier': factor})
        if name == 'original' and report['prediction_sha256'] != known['prediction_sha256']:
            raise RuntimeError('Unchanged B settings did not reproduce the original predictions')
        results[name] = report
        best = max(results.values(), key=lambda x: x['scores_times_100']['CIDEr'])
        atomic_json(output / 'comparison.json', {'best': best, 'results': results})
    atomic_json(output / 'progress.json', {'stage': 'complete', 'best': best['trial']})


if __name__ == '__main__':
    main()
