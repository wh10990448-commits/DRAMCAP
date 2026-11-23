"""HierarQ-inspired scene/entity temporal adapter with explicit 2-D relations.

The scene path preserves the previously validated VideoTemporalAdapter.  New
object and memory paths enter through small, trainable residual gates so an
experiment can be initialized from the CIDEr 71.29 checkpoint without
discarding its visual-language alignment.
"""
import math

import torch
from torch import nn

from .temporal import MemoryResampler, VideoTemporalAdapter


class ObjectRelationEncoder(nn.Module):
    """Encode detector category, box geometry, and pairwise direction tokens."""

    def __init__(self, dim=768, categories=91, heads=8, layers=2):
        super().__init__()
        self.category = nn.Embedding(categories, dim, padding_idx=0)
        self.absolute = nn.Sequential(
            nn.Linear(12, dim // 2), nn.GELU(), nn.LayerNorm(dim // 2),
            nn.Linear(dim // 2, dim),
        )
        self.relative = nn.Sequential(
            nn.Linear(9, dim // 2), nn.GELU(), nn.Linear(dim // 2, dim),
        )
        # 0=same/invalid; 1..8 are the eight image-plane sectors.
        self.direction = nn.Embedding(9, dim, padding_idx=0)
        self.query = nn.Linear(dim, dim, bias=False)
        self.key = nn.Linear(dim, dim, bias=False)
        self.value = nn.Linear(dim, dim, bias=False)
        self.relation_score = nn.Linear(dim, 1, bias=False)
        self.relation_scale = nn.Parameter(torch.tensor(0.05))
        layer = nn.TransformerEncoderLayer(
            dim, heads, dim * 2, dropout=0.1, batch_first=True, norm_first=True)
        self.frame_encoder = nn.TransformerEncoder(
            layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(dim)

    def forward(self, category_ids, absolute, relative, directions, valid):
        batch, frames, objects = category_ids.shape
        valid = valid.bool()
        base = self.category(category_ids.clamp(0, self.category.num_embeddings - 1))
        base = base + self.absolute(absolute.float())

        relation = self.relative(relative.float()) + self.direction(
            directions.clamp(0, self.direction.num_embeddings - 1))
        left = self.query(base)[:, :, :, None, :]
        right = self.key(base)[:, :, None, :, :]
        scores = self.relation_score(torch.tanh(relation + left + right)).squeeze(-1)
        pair_valid = valid[:, :, :, None] & valid[:, :, None, :]
        eye = torch.eye(objects, dtype=torch.bool, device=valid.device)
        pair_valid = pair_valid & ~eye[None, None]
        has_relation = pair_valid.any(-1)
        safe_pair = pair_valid.clone()
        # Keep softmax finite for isolated/padded objects.  The artificial
        # entry is removed again by the multiplication with ``pair_valid``.
        safe_pair[..., 0] |= ~has_relation
        weights = scores.masked_fill(~safe_pair, -1e4).softmax(-1)
        weights = weights * pair_valid
        neighbor = self.value(base)[:, :, None, :, :]
        context = (weights[..., None] * (relation + neighbor)).sum(-2)
        tokens = self.norm(base + self.relation_scale.tanh() * context)

        flat = tokens.reshape(batch * frames, objects, -1)
        flat_valid = valid.reshape(batch * frames, objects)
        present = flat_valid.any(-1)
        safe_mask = ~flat_valid.clone()
        safe_mask[~present, 0] = False
        flat = flat.masked_fill((~present)[:, None, None], 0)
        flat = self.frame_encoder(flat, src_key_padding_mask=safe_mask)
        flat = flat * flat_valid[:, :, None]
        return flat.reshape(batch, frames, objects, -1), valid


class EntityTemporalMemory(nn.Module):
    """Compress objects per frame, model a local temporal window, then resample."""

    def __init__(self, dim=768, heads=8, frame_queries=4, output_queries=8,
                 temporal_layers=2, local_radius=2):
        super().__init__()
        self.frame_queries = nn.Parameter(torch.randn(1, frame_queries, dim) * 0.02)
        self.frame_attention = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.frame_norm = nn.LayerNorm(dim)
        layer = nn.TransformerEncoderLayer(
            dim, heads, dim * 2, dropout=0.1, batch_first=True, norm_first=True)
        self.temporal = nn.TransformerEncoder(
            layer, temporal_layers, enable_nested_tensor=False)
        self.output_queries = nn.Parameter(torch.randn(1, output_queries, dim) * 0.02)
        self.output_attention = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.output_norm = nn.LayerNorm(dim)
        self.local_radius = local_radius

    def forward(self, objects, object_valid, time_embedding, frame_valid):
        batch, frames, count, dim = objects.shape
        flat = objects.reshape(batch * frames, count, dim)
        valid = object_valid.reshape(batch * frames, count)
        present = valid.any(-1)
        safe_mask = ~valid.clone()
        safe_mask[~present, 0] = False
        queries = self.frame_queries.expand(batch * frames, -1, -1)
        pooled, _ = self.frame_attention(
            queries, flat, flat, key_padding_mask=safe_mask, need_weights=False)
        pooled = self.frame_norm(pooled + queries)
        pooled = pooled * present[:, None, None]
        pooled = pooled.reshape(batch, frames, -1, dim)
        pooled = pooled + time_embedding[:, :, None, :]

        per_query = pooled.permute(0, 2, 1, 3).reshape(-1, frames, dim)
        temporal_valid = frame_valid.bool() & present.reshape(batch, frames)
        frame_mask = (~temporal_valid)[:, None, :].expand(
            -1, pooled.shape[2], -1).reshape(-1, frames)
        temporal_present = (~frame_mask).any(-1)
        frame_mask = frame_mask.clone()
        frame_mask[~temporal_present, 0] = False
        per_query = per_query.masked_fill((~temporal_present)[:, None, None], 0)
        index = torch.arange(frames, device=objects.device)
        local_mask = (index[:, None] - index[None, :]).abs() > self.local_radius
        encoded = torch.zeros_like(per_query)
        if bool(temporal_present.any()):
            active = self.temporal(
                per_query[temporal_present], mask=local_mask,
                src_key_padding_mask=frame_mask[temporal_present])
            # A padded query position outside the radius of every valid key
            # may be all-masked inside attention.  It is never a usable
            # output, so overwrite it rather than multiplying a possible NaN
            # by zero.
            active = active.masked_fill(
                frame_mask[temporal_present, :, None], 0)
            encoded[temporal_present] = active
        encoded = encoded.reshape(batch, pooled.shape[2], frames, dim).permute(0, 2, 1, 3)
        memory = encoded.reshape(batch, frames * pooled.shape[2], dim)
        memory_valid = temporal_valid[:, :, None].expand(
            -1, -1, pooled.shape[2]).reshape(batch, -1).bool()
        entity_present = memory_valid.any(-1)
        safe_memory_mask = ~memory_valid
        safe_memory_mask = safe_memory_mask.clone()
        safe_memory_mask[~entity_present, 0] = False
        q = self.output_queries.expand(batch, -1, -1)
        output, _ = self.output_attention(
            q, memory, memory, key_padding_mask=safe_memory_mask, need_weights=False)
        output = self.output_norm(output + q)
        output_valid = entity_present[:, None].expand(-1, q.shape[1])
        output = output.masked_fill(~output_valid[:, :, None], 0)
        return output, output_valid


class HierarchicalSpatialTemporalAdapter(nn.Module):
    """Validated scene stream plus spatial entity and compressed-memory streams."""

    def __init__(self, dim=768, heads=8):
        super().__init__()
        self.scene = VideoTemporalAdapter(dim=dim, layers=2, heads=heads)
        self.scene_memory = MemoryResampler(dim=dim, queries=32, heads=heads)
        self.object_encoder = ObjectRelationEncoder(dim=dim, heads=heads)
        self.entity_memory = EntityTemporalMemory(dim=dim, heads=heads)
        self.scene_from_entity = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.scene_norm = nn.LayerNorm(dim)
        self.name_norm = nn.LayerNorm(dim)
        # Exact-zero gates make a baseline checkpoint reproduce its old
        # forward path before the first update.  Gate gradients are non-zero,
        # then the corresponding branch starts learning on later updates.
        self.scene_memory_scale = nn.Parameter(torch.tensor(0.0))
        self.entity_scene_scale = nn.Parameter(torch.tensor(0.0))
        self.entity_name_scale = nn.Parameter(torch.tensor(0.0))

    def initialize_scene_from_baseline(self, state):
        """Load the complete adapter from the validated pre-HierarQ checkpoint."""
        self.scene.load_state_dict(state, strict=True)

    @staticmethod
    def _time_features(timestamps, frame_mask):
        low = timestamps.masked_fill(~frame_mask, float('inf')).amin(1, keepdim=True)
        high = timestamps.masked_fill(~frame_mask, float('-inf')).amax(1, keepdim=True)
        position = (timestamps - low) / (high - low).clamp_min(1e-6)
        return torch.stack([
            position, position.square(), torch.sin(math.pi * position),
            torch.cos(math.pi * position)], dim=-1).float()

    def forward(self, visual, names, attributes, name_mask, attribute_mask,
                timestamps, frame_mask, object_ids, object_geometry,
                object_relations, direction_ids, object_mask):
        frame_mask = frame_mask.bool()
        scene, names_out, attributes_out, name_valid, attribute_valid = self.scene(
            visual, names, attributes, name_mask, attribute_mask,
            timestamps, frame_mask)
        time_embedding = self.scene.time_mlp(
            self._time_features(timestamps.float(), frame_mask))

        scene_valid = frame_mask[:, :, None].expand(-1, -1, visual.shape[2])
        long_scene, long_valid = self.scene_memory(
            visual, scene_valid, time_embedding)
        memory_gate = self.scene_memory_scale.tanh()
        scene = scene + memory_gate * (long_scene - scene)

        object_mask = object_mask.bool() & frame_mask[:, :, None]
        objects, object_valid = self.object_encoder(
            object_ids, object_geometry, object_relations,
            direction_ids, object_mask)
        entities, entity_valid = self.entity_memory(
            objects, object_valid, time_embedding, frame_mask)
        entity_present = entity_valid.any(-1)
        safe_entity_mask = ~entity_valid
        safe_entity_mask = safe_entity_mask.clone()
        safe_entity_mask[~entity_present, 0] = False
        entity_context, _ = self.scene_from_entity(
            scene, entities, entities, key_padding_mask=safe_entity_mask,
            need_weights=False)
        entity_context = entity_context * entity_present[:, None, None]
        # Normalize only the residual branch.  With zero gates this module is
        # therefore bit-identical to the validated scene adapter, instead of
        # silently re-normalizing its already aligned output.
        scene = scene + self.entity_scene_scale.tanh() * self.scene_norm(
            entity_context)
        names_out = names_out + self.entity_name_scale.tanh() * self.name_norm(
            entities)
        name_valid = name_valid | entity_valid
        return scene, names_out, attributes_out, name_valid, attribute_valid

    def gate_values(self):
        return {
            'scene_memory': float(self.scene_memory_scale.detach().tanh()),
            'entity_scene': float(self.entity_scene_scale.detach().tanh()),
            'entity_name': float(self.entity_name_scale.detach().tanh()),
        }
