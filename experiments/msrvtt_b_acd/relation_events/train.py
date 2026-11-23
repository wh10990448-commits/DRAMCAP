"""Two-GPU all-20-reference training; B/C/D share one frozen 73.96 base."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import datetime
import json
import math
import os
from pathlib import Path
import random

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

from .data import EventDataset, collate_events
from .model import EventCaptioner


def atomic_json(path, value):
    temporary = Path(str(path) + '.' + str(os.getpid()) + '.tmp')
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


def atomic_save(path, value):
    temporary = Path(str(path) + '.' + str(os.getpid()) + '.tmp')
    torch.save(value, temporary); temporary.replace(path)


def move(batch, device):
    return {key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor)
            else value for key, value in batch.items()}


def main(args):
    rank = int(os.environ.get('RANK', 0)); world = int(os.environ.get('WORLD_SIZE', 1))
    local = int(os.environ.get('LOCAL_RANK', 0))
    torch.cuda.set_device(local); device = torch.device('cuda', local)
    torch.set_num_threads(4)
    if world > 1:
        dist.init_process_group('nccl')
    # Identical new module initialization in all variants and ranks.
    random.seed(args.seed); torch.manual_seed(args.seed)
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    dataset = EventDataset(args.split_root, args.cache, args.spatial_cache,
                           args.event_cache, 'train', all_references=True, limit=args.train_limit)
    validation_ids = {json.loads(line)['video_id'] for line in
                      (Path(args.split_root) / 'val.jsonl').read_text().splitlines() if line}
    if validation_ids & {row['video_id'] for row in dataset.rows}:
        raise ValueError('Train/validation identity overlap')
    if any(len(row['references']) != 20 for row in dataset.rows):
        raise ValueError('Expected all 20 references per MSR-VTT training video')
    sampler = DistributedSampler(dataset, world, rank, shuffle=True, seed=args.seed) if world > 1 else None
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler,
                        shuffle=sampler is None, num_workers=args.workers,
                        collate_fn=collate_events, pin_memory=True, generator=generator,
                        multiprocessing_context='spawn' if args.workers else None)
    baseline = torch.load(args.baseline, map_location='cpu', weights_only=True)
    if baseline['feature_fingerprint'] != dataset.metadata['fingerprint']:
        raise ValueError('Baseline/visual cache mismatch')
    if baseline['spatial_fingerprint'] != dataset.spatial_metadata['fingerprint']:
        raise ValueError('Baseline/spatial cache mismatch')
    model = EventCaptioner(str(Path(args.root) / 'models/vicuna-7b-v1.5'),
                           Path(args.root) / 'stage2_final.pt', args.variant,
                           args.order_weight).to(device)
    model.load_baseline(baseline)
    model.enable_gradient_checkpointing()
    trainable = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    if not trainable or any(not name.startswith('events.') for name, _ in trainable):
        raise RuntimeError('Only the new relation-event branch may be trainable')
    if rank == 0:
        print('TRAINABLE_EVENT_PARAMETERS', sum(p.numel() for _, p in trainable), flush=True)
    # Rank-specific stochasticity begins only after identical initialization.
    random.seed(args.seed + rank); torch.manual_seed(args.seed + rank)
    params = [p for p in model.events.parameters() if p.requires_grad]
    groups = [{'params': [p for name, p in model.events.named_parameters() if name != 'scale'], 'lr': args.lr},
              {'params': [model.events.scale], 'lr': args.gate_lr, 'weight_decay': 0.0}]
    optimizer = torch.optim.AdamW(groups, weight_decay=0.01)
    per_epoch = math.ceil(len(loader) / args.accumulation)
    total = args.epochs * per_epoch
    warmup = max(1, int(total * 0.05))
    def scale(step):
        if step < warmup:
            return (step + 1) / warmup
        fraction = min(1.0, (step - warmup) / max(1, total - warmup))
        return 0.1 + 0.45 * (1 + math.cos(math.pi * fraction))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, scale)
    start = 0; steps = 0
    if args.resume and (output / 'latest.pt').exists():
        state = torch.load(output / 'latest.pt', map_location='cpu', weights_only=True)
        for key in ('feature_fingerprint', 'spatial_fingerprint', 'event_fingerprint'):
            expected = {'feature_fingerprint': dataset.metadata['fingerprint'],
                        'spatial_fingerprint': dataset.spatial_metadata['fingerprint'],
                        'event_fingerprint': dataset.event_metadata['fingerprint']}[key]
            if state[key] != expected:
                raise ValueError('Resume provenance mismatch: ' + key)
        if state['args']['epochs'] != args.epochs or state['world_size'] != world:
            raise ValueError('Resume schedule/world size changed')
        model.load_adapted_state(state); optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler']); start = state['epoch']; steps = state['steps']
    wrapped = DistributedDataParallel(model, device_ids=[local], broadcast_buffers=False,
                                      find_unused_parameters=True) if world > 1 else model
    optimizer.zero_grad(set_to_none=True); torch.cuda.reset_peak_memory_stats()
    for epoch in range(start, args.epochs):
        random.seed(args.seed + epoch * world + rank)
        torch.manual_seed(args.seed + epoch * world + rank)
        generator.manual_seed(args.seed + epoch)
        if sampler:
            sampler.set_epoch(epoch)
        wrapped.train(); totals = torch.zeros(4, dtype=torch.float64, device=device)
        for index, batch in enumerate(loader):
            boundary = (index + 1) % args.accumulation == 0 or index + 1 == len(loader)
            group_size = min(args.accumulation, len(loader) - index // args.accumulation * args.accumulation)
            context = wrapped.no_sync() if world > 1 and not boundary else nullcontext()
            with context, torch.autocast('cuda', dtype=torch.bfloat16):
                loss, count = wrapped(move(batch, device))
                if not torch.isfinite(loss):
                    raise RuntimeError('Non-finite event training loss')
                (loss / group_size).backward()
            totals[0] += loss.detach().double() * count
            totals[1] += count; totals[2] += len(batch['captions'])
            totals[3] += model.order_statistics.get('eligible', 0)
            if not boundary:
                continue
            norm = torch.nn.utils.clip_grad_norm_(params, 1.0)
            if not torch.isfinite(norm):
                raise RuntimeError('Non-finite event gradients')
            optimizer.step(); scheduler.step(); optimizer.zero_grad(set_to_none=True)
            steps += 1
            if rank == 0 and (steps == 1 or steps % 25 == 0):
                atomic_json(output / 'progress.json', {
                    'variant': args.variant, 'epoch': epoch + 1, 'epochs': args.epochs,
                    'steps': steps, 'steps_per_epoch': per_epoch, 'total_steps': total,
                    'caption_pairs_per_epoch': len(dataset), 'training_videos': len(dataset.rows),
                    'loss': float(loss.detach()), 'grad_norm': float(norm),
                    'event_gate': float(model.events.scale.detach().tanh()),
                    'effective_batch': args.batch_size * args.accumulation * world,
                    'learning_rate': optimizer.param_groups[0]['lr'],
                    'order_eligible_local': int(totals[3]),
                    'peak_gib': torch.cuda.max_memory_allocated() / 2**30,
                    'utc': datetime.datetime.now(datetime.timezone.utc).isoformat()})
            if args.max_steps and steps >= args.max_steps:
                break
        if world > 1:
            dist.all_reduce(totals)
        if rank == 0:
            state = {**model.adapted_state(), 'epoch': epoch + 1, 'steps': steps,
                     'args': vars(args), 'world_size': world,
                     'stage': 'all_20_caption_relation_events',
                     'baseline_checkpoint': args.baseline,
                     'baseline_validation_cider': 73.95684838876862,
                     'feature_fingerprint': dataset.metadata['fingerprint'],
                     'spatial_fingerprint': dataset.spatial_metadata['fingerprint'],
                     'event_fingerprint': dataset.event_metadata['fingerprint'],
                     'image_checkpoint_sha256': baseline['image_checkpoint_sha256'],
                     'train_objective': float(totals[0] / totals[1].clamp_min(1)),
                     'caption_pairs_presented': int(totals[2]), 'order_eligible': int(totals[3]),
                     'test_set_used_for_training_or_selection': False}
            atomic_save(output / f'epoch_{epoch + 1:03d}.pt', state)
            atomic_save(output / 'latest.pt', {**state, 'optimizer': optimizer.state_dict(),
                                              'scheduler': scheduler.state_dict()})
            print('EVENT_EPOCH_COMPLETE', args.variant, epoch + 1, state['train_objective'], flush=True)
        if world > 1:
            dist.barrier()
        if args.max_steps and steps >= args.max_steps:
            break
        if args.stop_after_epoch and epoch + 1 >= args.stop_after_epoch:
            break
    if world > 1:
        dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default='/root/autodl-tmp/DRAMCap_video')
    parser.add_argument('--split-root', required=True)
    parser.add_argument('--cache', required=True)
    parser.add_argument('--spatial-cache', required=True)
    parser.add_argument('--event-cache', required=True)
    parser.add_argument('--baseline', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--variant', choices=['B', 'C', 'D'], required=True)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--accumulation', type=int, default=3)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--gate-lr', type=float, default=2e-5)
    parser.add_argument('--order-weight', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--train-limit', type=int, default=0)
    parser.add_argument('--max-steps', type=int, default=0)
    parser.add_argument('--stop-after-epoch', type=int, default=0)
    main(parser.parse_args())
