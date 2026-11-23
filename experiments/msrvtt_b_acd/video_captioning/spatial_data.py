"""Dataset overlay that joins validated EVA caches with detector geometry."""
import json
from pathlib import Path

import torch

from .data import CachedVideoDataset


SPATIAL_KEYS = [
    'object_ids', 'object_geometry', 'object_relations',
    'direction_ids', 'object_mask',
]


class SpatialCachedVideoDataset(CachedVideoDataset):
    def __init__(self, split_path, cache_root, spatial_root, split,
                 seed=42, limit=0):
        super().__init__(split_path, cache_root, split, seed, limit)
        self.spatial_root = Path(spatial_root)
        self.spatial_metadata = json.loads(
            (self.spatial_root / 'cache_metadata.json').read_text())
        if self.spatial_metadata['source_feature_fingerprint'] != self.metadata['fingerprint']:
            raise ValueError('Spatial sidecar and EVA feature cache provenance differ')
        missing = [r['video_id'] for r in self.rows
                   if not (self.spatial_root / split / (r['video_id'] + '.pt')).exists()]
        if missing:
            raise FileNotFoundError(
                f'Missing {len(missing)} spatial sidecars; first: {missing[:3]}')

    def __getitem__(self, index):
        feature, caption = super().__getitem__(index)
        row = self.rows[index]
        spatial = torch.load(
            self.spatial_root / self.split / (row['video_id'] + '.pt'),
            map_location='cpu', weights_only=True)
        if spatial['video_id'] != row['video_id'] or spatial['split'] != self.split:
            raise ValueError('Spatial sidecar identity/split mismatch')
        if spatial['cache_fingerprint'] != self.spatial_metadata['fingerprint']:
            raise ValueError('Stale spatial sidecar')
        if not torch.equal(spatial['frame_mask'], feature['frame_mask']):
            raise ValueError('Spatial and EVA frame masks are not aligned')
        if not torch.allclose(
                spatial['timestamps'].float(), feature['timestamps'].float(), atol=1e-4, rtol=0):
            raise ValueError('Spatial and EVA timestamps are not aligned')
        merged = dict(feature)
        for key in SPATIAL_KEYS:
            merged[key] = spatial[key]
        return merged, caption, row['references']


def collate_spatial_features(items):
    tensor_keys = [
        'visual', 'names', 'attributes', 'name_mask', 'attribute_mask',
        'timestamps', 'frame_mask', *SPATIAL_KEYS,
    ]
    result = {key: torch.stack([item[0][key] for item in items])
              for key in tensor_keys}
    result['video_ids'] = [item[0]['video_id'] for item in items]
    result['captions'] = [item[1] for item in items]
    result['references'] = [item[2] for item in items]
    return result
