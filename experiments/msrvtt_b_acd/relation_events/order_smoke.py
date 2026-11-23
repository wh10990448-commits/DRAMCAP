"""Exercise D's additional forwards/backwards on real eligible training pairs."""
import argparse
import json
from pathlib import Path

import torch
from .branch import eligible_order_caption
from .data import EventDataset, collate_events
from .model import EventCaptioner
from .train import atomic_json, move


def main(args):
    torch.set_num_threads(4)
    torch.manual_seed(42)
    dataset = EventDataset(args.split_root, args.cache, args.spatial_cache,
                           args.event_cache, 'train', all_references=True,
                           limit=args.train_limit)
    candidates = []
    offset = 0
    for row in dataset.rows:
        event = torch.load(Path(args.event_cache)/'train'/(row['video_id']+'.pt'),
                           map_location='cpu', weights_only=True)
        evidence = event['event_reliability'][..., :4].float().prod(-1).sqrt()
        if bool((event['event_mask'] & (evidence > .4) & (event['event_change'] > .1)).any()):
            candidates.extend(offset+i for i,c in enumerate(row['references']) if eligible_order_caption(c))
        offset += len(row['references'])
        if len(candidates) >= args.batch_size:
            break
    if not candidates:
        raise RuntimeError('No genuine D-eligible motion examples in smoke data')
    items = [dataset[candidates[i % len(candidates)]] for i in range(args.batch_size)]
    batch = move(collate_events(items), 'cuda:0')
    model = EventCaptioner(str(Path(args.root)/'models/vicuna-7b-v1.5'),
                           Path(args.root)/'stage2_final.pt', 'D').cuda()
    model.load_baseline(torch.load(args.baseline, map_location='cpu', weights_only=True))
    model.enable_gradient_checkpointing(); model.train()
    optimizer = torch.optim.AdamW(model.events.parameters(), lr=2e-5)
    torch.cuda.reset_peak_memory_stats()
    records = []
    for step in range(2):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            loss, _ = model(batch)
        loss.backward()
        if model.order_statistics['eligible'] != args.batch_size:
            raise RuntimeError('D order-ranking path was not actually exercised')
        gate_gradient = model.events.scale.grad
        if not bool(torch.isfinite(gate_gradient)) or float(gate_gradient) == 0:
            raise RuntimeError('No finite nonzero event-gate gradient through frozen LLM')
        if step and not bool(model.events.appearance.weight.grad.abs().sum() > 0):
            raise RuntimeError('No gradient reached the new event representation')
        if any(p.grad is not None for p in model.adapter.parameters()) or any(
                p.grad is not None for p in model.decoder.parameters()):
            raise RuntimeError('A frozen baseline parameter received a gradient')
        records.append({'step': step+1, 'loss':float(loss.detach()),
                        'order':dict(model.order_statistics),
                        'gate_gradient':float(gate_gradient),
                        'peak_gib':torch.cuda.max_memory_allocated()/2**30})
        optimizer.step()
    report = {'passed':True,'batch_size':args.batch_size,'real_eligible_pairs':len(candidates),
              'steps':records,'captions':batch['captions'],'test_set_used':False,
              'scientific_evaluation':False}
    atomic_json(args.output, report)
    print('REAL_ORDER_SMOKE',json.dumps(report),flush=True)
    if records[-1]['peak_gib'] > 21.5:
        raise RuntimeError('out of memory safety budget: reserve memory for longer captions')


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    for name in ('root','split-root','cache','spatial-cache','event-cache','baseline','output'):
        parser.add_argument('--'+name,required=True)
    parser.add_argument('--batch-size',type=int,default=24)
    parser.add_argument('--train-limit',type=int,default=0)
    main(parser.parse_args())
