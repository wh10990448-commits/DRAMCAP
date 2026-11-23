"""Synthetic two-3090 gradient/masking checks; NOT an MSR-VTT training run."""
import json
import os

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from .branch import RelationEventBranch


def main():
    local = int(os.environ['LOCAL_RANK']); rank = int(os.environ['RANK'])
    torch.cuda.set_device(local); torch.set_num_threads(2)
    dist.init_process_group('nccl')
    records = []
    for variant in ('B', 'C', 'D'):
        torch.manual_seed(42)
        module = RelationEventBranch(variant).cuda()
        wrapped = DistributedDataParallel(module, device_ids=[local], find_unused_parameters=True)
        optimizer = torch.optim.AdamW(module.parameters(), lr=1e-3)
        torch.manual_seed(100 + rank)
        batch = {'event_appearance': torch.randn(2, 504, 2048, device='cuda'),
                 'event_geometry': torch.randn(2, 504, 32, device='cuda'),
                 'event_reliability': torch.full((2, 504, 5), .9, device='cuda'),
                 'event_change': torch.ones(2, 504, device='cuda'),
                 'event_mask': torch.ones(2, 504, dtype=torch.bool, device='cuda')}
        batch['event_mask'][1] = False
        scene = torch.randn(2, 32, 768, device='cuda')
        for _ in range(3):
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                output = wrapped(scene, batch)
                loss = output.float().square().mean()
            if not torch.isfinite(loss):
                raise RuntimeError('Non-finite synthetic loss')
            torch.testing.assert_close(output[1], scene[1], rtol=0, atol=0)
            loss.backward()
            if module.scale.grad is None or not torch.isfinite(module.scale.grad):
                raise RuntimeError('Missing/non-finite residual gate gradient')
            optimizer.step()
        gate = module.scale.detach().float()
        gathered = [torch.empty_like(gate) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered, gate)
        for other in gathered:
            torch.testing.assert_close(other, gate, rtol=0, atol=0)
        records.append({'variant': variant, 'finite': True, 'exact_empty_fallback': True,
                        'ddp_gate_identical': True, 'trained_gate': float(gate)})
    if rank == 0:
        print('SYNTHETIC_DDP_COMPONENT_SMOKE_PASSED', json.dumps(records), flush=True)
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
