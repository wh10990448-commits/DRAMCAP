"""B: relation changes. C: reliability and fixed-budget event memory.

D has exactly C's inference path and adds selected order supervision at train
time. All experiments start from the same baseline, never each other's best.
"""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class RelationEventBranch(nn.Module):
    def __init__(self, variant='B', appearance_dim=2048, dim=256, output_dim=768,
                 heads=4, memory_slots=16):
        super().__init__()
        if variant not in ('B', 'C', 'D'):
            raise ValueError('Expected variant B, C or D')
        self.variant = variant
        self.memory_slots = memory_slots
        self.appearance = nn.Linear(appearance_dim, dim)
        self.geometry = nn.Sequential(nn.Linear(32, dim), nn.GELU(), nn.Linear(dim, dim))
        self.norm = nn.LayerNorm(dim)
        self.reliability_gate = nn.Sequential(nn.Linear(5, 32), nn.GELU(), nn.Linear(32, 1))
        self.memory_queries = nn.Parameter(torch.randn(1, memory_slots, dim) * 0.02)
        self.memory_attention = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.memory_norm = nn.LayerNorm(dim)
        self.scene_query = nn.Linear(output_dim, dim)
        self.scene_attention = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.output = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, output_dim))
        self.scale = nn.Parameter(torch.tensor(0.0))

    def forward(self, scene, batch):
        mask = batch['event_mask'].bool()
        present = mask.any(1)
        events = self.norm(self.appearance(batch['event_appearance'].float())
                           + self.geometry(batch['event_geometry'].float()))
        reliability = batch['event_reliability'].float()
        video_quality = scene.new_ones(scene.shape[0], dtype=torch.float32)
        if self.variant in ('C', 'D'):
            # A low detector score is different from an ambiguous association.
            # Both gate the branch; neither is called a calibrated probability.
            evidence = reliability[:, :, :4].clamp(0, 1).prod(-1).sqrt()
            learned = self.reliability_gate(reliability).squeeze(-1).sigmoid()
            quality = evidence * learned
            mask = mask & (quality > 0.05)
            present = mask.any(1)
            video_quality = quality.masked_fill(~mask, 0).amax(1)
            # Rank by change only after discounting unreliable correspondences.
            priority = quality * (0.1 + torch.log1p(batch['event_change'].float()))
            priority = priority.masked_fill(~mask, -1e4)
            count = min(self.memory_slots * 4, mask.shape[1])
            indices = priority.topk(count, dim=1).indices
            events = events.gather(1, indices[:, :, None].expand(-1, -1, events.shape[-1]))
            mask = mask.gather(1, indices)
            selected_quality = quality.gather(1, indices)
            events = events * selected_quality[:, :, None]
        safe_mask = ~mask.clone()
        safe_mask[~present, 0] = False
        events = events.masked_fill((~present)[:, None, None], 0)
        if self.variant in ('C', 'D'):
            query = self.memory_queries.expand(scene.shape[0], -1, -1)
            memory, _ = self.memory_attention(
                query, events, events, key_padding_mask=safe_mask, need_weights=False)
            events = self.memory_norm(query + memory)
            safe_mask = torch.zeros(events.shape[:2], dtype=torch.bool, device=events.device)
        context, _ = self.scene_attention(
            self.scene_query(scene.float()), events, events,
            key_padding_mask=safe_mask, need_weights=False)
        residual = self.output(context).masked_fill((~present)[:, None, None], 0)
        residual = residual * video_quality[:, None, None]
        return scene + self.scale.tanh() * residual


def eligible_order_caption(caption):
    """Conservative textual evidence; generic action captions are not negatives.

No generated inverse captions or blanket 'all reversed videos are wrong'.
The relation-change sidecar must also contain a reliable observed transition.
"""
    import re
    # "eats out of a bowl" describes a source, and "then sings" need not
    # depend on the box geometry. Require an actual directed-motion phrase.
    text = caption.lower()
    if re.search(r'\b(ready to|preparing to|about to)\b', text):
        return False
    return bool(re.search(
        r'\b(enters?|entering|exits?|exiting|approaches?|approaching)\b|'
        r'\b(moves?|moving|walks?|walking|runs?|running|goes|going|'
        r'drives?|driving|flies|flying|swims?|swimming|steps?|stepping|'
        r'jumps?|jumping|comes?|coming)\s+(?:\w+\s+){0,2}'
        r'(away from|towards?|into|out of)\b', text))


def reverse_event_dynamics(batch):
    """Temporal counterfactual for the event stream, NOT a reversed raw video.

Appearance and object identities stay fixed. Current/previous geometry are
exchanged, the rate changes sign, and time is reflected. In the paper/report
this must be called an event-order counterfactual, not full video reversal.
"""
    result = dict(batch)
    geometry = batch['event_geometry'].clone()
    geometry[..., :9] = batch['event_geometry'][..., 9:18]
    geometry[..., 9:18] = batch['event_geometry'][..., :9]
    geometry[..., 18:27] = -batch['event_geometry'][..., 18:27]
    time = 1 - batch['event_geometry'][..., 27]
    geometry[..., 27] = time
    geometry[..., 28] = time.square()
    geometry[..., 29] = (time * torch.pi).sin()
    geometry[..., 30] = (time * torch.pi).cos()
    result['event_geometry'] = geometry
    return result


def order_ranking_loss(positive, counterfactual, eligible, margin=0.05):
    values = F.relu(margin - positive + counterfactual)
    eligible = eligible.to(values.dtype)
    return (values * eligible).sum() / eligible.sum().clamp_min(1)
