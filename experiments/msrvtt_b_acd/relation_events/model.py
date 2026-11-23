"""Wrap the original spatial captioner without renaming its checkpoint keys."""
from __future__ import annotations

import torch

from video_captioning.model import VideoCaptioner as BaselineCaptioner
from .branch import (RelationEventBranch, eligible_order_caption,
                     order_ranking_loss, reverse_event_dynamics)


class EventCaptioner(BaselineCaptioner):
    def __init__(self, model_dir, image_checkpoint, variant='B', order_weight=0.1):
        super().__init__(model_dir, image_checkpoint, train_projector=False)
        self.adapter.requires_grad_(False)
        self.decoder.requires_grad_(False)
        self.events = RelationEventBranch(variant=variant)
        self.variant = variant
        self.order_weight = order_weight
        self.order_statistics = {}

    def train(self, mode=True):
        super().train(mode)
        self.decoder.eval()
        # Hugging Face only activates checkpointing in training mode. The LLM
        # weights stay frozen; Vicuna's configured dropout is zero.
        self.decoder.llm.train(mode)
        self.adapter.eval()
        self.events.train(mode)
        return self

    def tokens(self, batch):
        scene, names, attributes, name_valid, attribute_valid = super().tokens(batch)
        scene = self.events(scene, batch)
        return scene, names, attributes, name_valid, attribute_valid

    def load_baseline(self, state):
        super().load_adapted_state(state)

    def adapted_state(self):
        return {**super().adapted_state(), 'relation_events': self.events.state_dict(),
                'variant': self.variant, 'event_schema': 'relation-events-v1'}

    def load_adapted_state(self, state):
        self.load_baseline(state)
        if 'relation_events' in state:
            if state['variant'] != self.variant:
                raise ValueError('Cannot silently load a different ablation variant')
            self.events.load_state_dict(state['relation_events'], strict=True)

    def caption_logprob(self, batch, captions):
        prefix, prefix_mask, length = self._prefix(batch)
        ids, mask = self._caption_tensors(captions, prefix.device)
        attention = torch.cat([prefix_mask, mask], dim=1)
        positions = attention.long().cumsum(-1) - 1
        positions.masked_fill_(attention == 0, 1)
        outputs = self.decoder.llm(
            inputs_embeds=torch.cat([prefix, self.decoder._get_embed_fn()(ids)], dim=1),
            attention_mask=attention, position_ids=positions, use_cache=False, return_dict=True)
        logits = outputs.logits[:, length - 1:-1].float()
        logprob = logits.log_softmax(-1).gather(-1, ids[:, :, None]).squeeze(-1)
        return (logprob * mask).sum(-1) / mask.sum(-1).clamp_min(1)

    def forward(self, batch):
        loss, count = super().forward(batch)
        self.order_statistics = {'eligible': 0, 'pairs': len(batch['captions'])}
        if self.variant != 'D' or not self.training:
            return loss, count
        # Caption syntax alone is insufficient: require a reliable transition.
        evidence = batch['event_reliability'][..., :4].float().prod(-1).sqrt()
        has_transition = (batch['event_mask'].bool() & (evidence > 0.4)
                          & (batch['event_change'] > 0.1)).any(1)
        eligible = torch.tensor([eligible_order_caption(s) for s in batch['captions']],
                                device=has_transition.device) & has_transition
        indices = eligible.nonzero().flatten()
        self.order_statistics['eligible'] = len(indices)
        if not len(indices):
            return loss, count
        subset = {key: (value[indices] if isinstance(value, torch.Tensor)
                        else [value[int(i)] for i in indices])
                  for key, value in batch.items()}
        positive = self.caption_logprob(subset, subset['captions'])
        negative = self.caption_logprob(reverse_event_dynamics(subset), subset['captions'])
        order = order_ranking_loss(positive, negative, torch.ones_like(positive))
        self.order_statistics['loss'] = float(order.detach())
        return loss + self.order_weight * order, count
