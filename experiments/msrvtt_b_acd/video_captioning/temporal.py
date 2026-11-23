"""Temporal fusion of per-frame ROI visual tokens and retrieved memories."""
import math

import torch
from torch import nn


class MemoryResampler(nn.Module):
    def __init__(self, dim=768, queries=8, heads=8):
        super().__init__()
        self.queries = nn.Parameter(torch.randn(1, queries, dim)*0.02)
        self.norm = nn.LayerNorm(dim)
        self.attention = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.output_norm = nn.LayerNorm(dim)

    def forward(self, tokens, valid, time_embedding):
        batch, frames, length, dim = tokens.shape
        present = valid.reshape(batch, -1).any(1)
        kv = (tokens.float()+time_embedding[:, :, None, :]).reshape(batch, frames*length, dim)
        mask = ~valid.reshape(batch, -1)
        # MultiheadAttention cannot consume a row whose every key is masked.
        mask = mask.clone(); mask[~present, 0] = False
        kv = kv.masked_fill((~present)[:, None, None], 0)
        q = self.queries.expand(batch, -1, -1)
        output, _ = self.attention(q, self.norm(kv), self.norm(kv), key_padding_mask=mask, need_weights=False)
        output = self.output_norm(output+q)*present[:, None, None]
        return output, present[:, None].expand(-1, q.shape[1])


class VideoTemporalAdapter(nn.Module):
    def __init__(self, dim=768, layers=2, heads=8, dropout=0.1):
        super().__init__()
        self.time_mlp = nn.Sequential(nn.Linear(4, dim), nn.GELU(), nn.Linear(dim, dim))
        layer = nn.TransformerEncoderLayer(dim, heads, dim*2, dropout=dropout,
                                           batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.score = nn.Linear(dim, 1)
        self.residual_scale = nn.Parameter(torch.tensor(0.01))
        self.name_pool = MemoryResampler(dim, 8, heads)
        self.attribute_pool = MemoryResampler(dim, 8, heads)

    def forward(self, visual, names, attributes, name_mask, attribute_mask, timestamps, frame_mask):
        batch, frames, queries, dim = visual.shape
        frame_mask = frame_mask.bool()
        timestamps = timestamps.float()
        if not bool(frame_mask.any(1).all()):
            raise ValueError('Every video needs at least one valid frame')
        if not bool(torch.isfinite(timestamps[frame_mask]).all()):
            raise ValueError('Valid frames need finite timestamps')
        low = timestamps.masked_fill(~frame_mask, float('inf')).amin(1, keepdim=True)
        high = timestamps.masked_fill(~frame_mask, float('-inf')).amax(1, keepdim=True)
        position = (timestamps-low)/(high-low).clamp_min(1e-6)
        features = torch.stack([position, position.square(), torch.sin(math.pi*position),
                                torch.cos(math.pi*position)], dim=-1).float()
        time_embedding = self.time_mlp(features)
        tokens = (visual.float()+time_embedding[:, :, None, :]).permute(0, 2, 1, 3).reshape(batch*queries, frames, dim)
        mask = (~frame_mask)[:, None, :].expand(-1, queries, -1).reshape(batch*queries, frames)
        encoded = self.encoder(tokens, src_key_padding_mask=mask)
        weights = self.score(encoded).squeeze(-1).masked_fill(mask, float('-inf')).softmax(1)
        temporal = (encoded*weights[:, :, None]).sum(1).reshape(batch, queries, dim)
        base = (visual.float()*frame_mask[:, :, None, None]).sum(1)/frame_mask.sum(1)[:, None, None]
        output = base+self.residual_scale.tanh()*(temporal-base)
        names, name_valid = self.name_pool(names, name_mask.bool() & frame_mask[:, :, None], time_embedding)
        attributes, attribute_valid = self.attribute_pool(attributes, attribute_mask.bool() & frame_mask[:, :, None], time_embedding)
        return output, names, attributes, name_valid, attribute_valid
