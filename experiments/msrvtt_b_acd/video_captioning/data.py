"""Timestamp-aware video sampling and split-checked feature datasets."""
import hashlib
import json
from pathlib import Path

import torch
from torch.utils.data import Dataset


def sample_video(path, count=8):
    import av
    if count < 1:
        raise ValueError('Frame count must be positive')
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = 'AUTO'
        stream.thread_count = 2  # Bound FFmpeg threads on shared/container hosts.
        origin = float((stream.start_time or 0)*stream.time_base)
        duration = float(stream.duration*stream.time_base) if stream.duration else None
        if not duration or duration <= 0:
            times = [float(f.pts*f.time_base) for f in container.decode(stream) if f.pts is not None]
            if not times:
                raise ValueError('Video has no timestamped frames: '+str(path))
            origin = times[0]
            duration = max(times[-1]-origin, 1e-6)
            container.seek(0, stream=stream)
        targets = [origin+duration*(i+0.5)/count for i in range(count)]
        chosen = []; previous = None; last_time = None; index = 0
        for frame in container.decode(stream):
            if frame.pts is None:
                raise ValueError('Missing frame timestamp: '+str(path))
            timestamp = float(frame.pts*frame.time_base)
            if last_time is not None and timestamp < last_time-1e-6:
                raise ValueError('Non-monotonic display timestamps: '+str(path))
            while index < count and timestamp >= targets[index]:
                candidate = (timestamp, frame)
                if previous is not None and abs(previous[0]-targets[index]) <= abs(timestamp-targets[index]):
                    candidate = previous
                chosen.append((candidate[0]-origin, candidate[1].to_image().convert('RGB')))
                index += 1
            previous = (timestamp, frame); last_time = timestamp
            if index == count:
                break
        if previous is None:
            raise ValueError('No decoded frames: '+str(path))
        while len(chosen) < count:
            chosen.append((previous[0]-origin, previous[1].to_image().convert('RGB')))
    timestamps = torch.tensor([x[0] for x in chosen], dtype=torch.float32)
    mask = torch.ones(count, dtype=torch.bool)
    for i in range(1, count):
        mask[i] = abs(float(timestamps[i]-timestamps[i-1])) > 1e-6
    return [x[1] for x in chosen], timestamps, mask


def load_split(path, expected_split):
    rows = [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]
    ids = [r['video_id'] for r in rows]
    if len(ids) != len(set(ids)):
        raise ValueError('Repeated video IDs in '+str(path))
    if any(r['split'] != expected_split for r in rows):
        raise ValueError('Split mismatch: '+str(path))
    if any(not r['references'] for r in rows):
        raise ValueError('Missing caption references')
    return rows


class CachedVideoDataset(Dataset):
    def __init__(self, split_path, cache_root, split, seed=42, limit=0):
        self.rows = load_split(split_path, split)
        if limit:
            self.rows = self.rows[:limit]
        self.cache_root = Path(cache_root)
        self.split = split; self.seed = seed; self.epoch = 0
        self.metadata = json.loads((self.cache_root/'cache_metadata.json').read_text())
        if not self.metadata.get('image_checkpoint_sha256'):
            raise ValueError('Cache lacks original-checkpoint provenance')
        missing = [r['video_id'] for r in self.rows if not (self.cache_root/split/(r['video_id']+'.pt')).exists()]
        if missing:
            raise FileNotFoundError(f'Missing {len(missing)} feature files; first: {missing[:3]}')

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        feature = torch.load(self.cache_root/self.split/(row['video_id']+'.pt'), map_location='cpu', weights_only=True)
        if feature['video_id'] != row['video_id'] or feature['split'] != self.split:
            raise ValueError('Cached sample identity/split mismatch')
        if feature['cache_fingerprint'] not in {self.metadata['fingerprint'], *self.metadata.get('compatible_fingerprints', [])}:
            raise ValueError('Stale feature cache')
        reference = 0
        if self.split == 'train':
            seed = f'{self.seed}:{self.epoch}:{row["video_id"]}'.encode()
            reference = int.from_bytes(hashlib.sha256(seed).digest()[:8], 'big') % len(row['references'])
        return feature, row['references'][reference]


def collate_features(items):
    result = {}
    tensor_keys = ['visual', 'names', 'attributes', 'name_mask', 'attribute_mask', 'timestamps', 'frame_mask']
    for key in tensor_keys:
        result[key] = torch.stack([x[0][key] for x in items])
    result['video_ids'] = [x[0]['video_id'] for x in items]
    result['captions'] = [x[1] for x in items]
    return result
