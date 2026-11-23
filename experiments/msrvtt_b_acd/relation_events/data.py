"""All-reference dataset with strict baseline/cache/sidecar provenance."""
from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import Dataset
from video_captioning.spatial_data import SpatialCachedVideoDataset, collate_spatial_features

EVENT_KEYS = ['event_appearance', 'event_geometry', 'event_reliability',
              'event_change', 'event_mask', 'event_frame']


class EventDataset(Dataset):
    def __init__(self, splits, cache, spatial, events, split, all_references=False, limit=0):
        self.base = SpatialCachedVideoDataset(Path(splits) / (split + '.jsonl'),
                                             cache, spatial, split, limit=limit)
        self.event_root = Path(events)
        self.event_metadata = json.loads((self.event_root / 'cache_metadata.json').read_text())
        if self.event_metadata['spatial_fingerprint'] != self.base.spatial_metadata['fingerprint']:
            raise ValueError('Event and spatial cache provenance differs')
        if self.event_metadata['feature_fingerprint'] != self.base.metadata['fingerprint']:
            raise ValueError('Event and visual cache provenance differs')
        self.rows = self.base.rows
        self.metadata = self.base.metadata
        self.spatial_metadata = self.base.spatial_metadata
        self.split = split
        self.pairs = [(i, j) for i, row in enumerate(self.rows)
                      for j in range(len(row['references']) if all_references else 1)]
        for row in self.rows:
            if not (self.event_root / split / (row['video_id'] + '.pt')).is_file():
                raise FileNotFoundError('Missing event sidecar: ' + row['video_id'])

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        video_index, reference_index = self.pairs[index]
        feature, _, references = self.base[video_index]
        event = torch.load(self.event_root / self.split / (feature['video_id'] + '.pt'),
                           map_location='cpu', weights_only=True)
        if event['video_id'] != feature['video_id'] or event['split'] != self.split:
            raise ValueError('Event identity/split mismatch')
        if event['cache_fingerprint'] != self.event_metadata['fingerprint']:
            raise ValueError('Stale event sidecar')
        if event['source_video_sha256'] != feature['source_video_sha256']:
            raise ValueError('Event and video source differ')
        if not torch.allclose(event['timestamps'].float(), feature['timestamps'].float(), atol=1e-4, rtol=0):
            raise ValueError('Event timestamps differ from baseline')
        return {**feature, **{k: event[k] for k in EVENT_KEYS}}, references[reference_index], references


def collate_events(items):
    result = collate_spatial_features(items)
    length = max(item[0]['event_mask'].shape[0] for item in items)
    for key in EVENT_KEYS:
        values = []
        for item in items:
            value = item[0][key]
            padded = value.new_full((length, *value.shape[1:]), -1 if key == 'event_frame' else 0)
            padded[:value.shape[0]] = value
            values.append(padded)
        result[key] = torch.stack(values)
    return result
