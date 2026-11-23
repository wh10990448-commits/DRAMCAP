"""Offline association and relation changes; detector indices are not track IDs.

Associations are hypotheses, not ground truth. The margin is a reliability
feature, not a calibrated probability. Missing links and shot boundaries never
produce a motion target. All variants consume the exact same sidecars.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
try:
    from scipy.optimize import linear_sum_assignment
except ImportError:
    def linear_sum_assignment(cost):
        """Exact rectangular Hungarian assignment for our <=9-by-9 case.

        Standard shortest augmenting paths with row/column potentials. Keep a
        dependency-free fallback so an unavailable package mirror cannot block
        preprocessing. Returns the same row/column contract as scipy.
        """
        values = cost.tolist()
        rows = len(values); columns = len(values[0]) if rows else 0
        if not rows or not columns:
            return [], []
        transposed = rows > columns
        if transposed:
            values = [list(item) for item in zip(*values)]
            rows, columns = columns, rows
        u = [0.] * (rows + 1); v = [0.] * (columns + 1)
        owner = [0] * (columns + 1); predecessor = [0] * (columns + 1)
        for row in range(1, rows + 1):
            owner[0] = row; column = 0
            minimum = [float('inf')] * (columns + 1)
            visited = [False] * (columns + 1)
            while True:
                visited[column] = True; active = owner[column]
                delta = float('inf'); next_column = 0
                for j in range(1, columns + 1):
                    if visited[j]:
                        continue
                    reduced = values[active - 1][j - 1] - u[active] - v[j]
                    if reduced < minimum[j]:
                        minimum[j] = reduced; predecessor[j] = column
                    if minimum[j] < delta:
                        delta = minimum[j]; next_column = j
                for j in range(columns + 1):
                    if visited[j]:
                        u[owner[j]] += delta; v[j] -= delta
                    else:
                        minimum[j] -= delta
                column = next_column
                if owner[column] == 0:
                    break
            while column:
                old = predecessor[column]; owner[column] = owner[old]; column = old
        pairs = [(owner[j] - 1, j - 1) for j in range(1, columns + 1) if owner[j]]
        if transposed:
            pairs = [(b, a) for a, b in pairs]
        pairs.sort()
        return [a for a, _ in pairs], [b for _, b in pairs]


def iou_matrix(first, second):
    low = torch.maximum(first[:, None, :2], second[None, :, :2])
    high = torch.minimum(first[:, None, 2:], second[None, :, 2:])
    intersection = (high - low).clamp_min(0).prod(-1)
    areas_a = (first[:, 2:] - first[:, :2]).clamp_min(0).prod(-1)
    areas_b = (second[:, 2:] - second[:, :2]).clamp_min(0).prod(-1)
    return intersection / (areas_a[:, None] + areas_b[None] - intersection).clamp_min(1e-6)


def associate(categories, geometry, appearance, valid, timestamps, shot_starts):
    """Return predecessor detection indices and association reliability [T,N]."""
    frames, count = categories.shape
    previous = torch.full((frames, count), -1, dtype=torch.long)
    reliability = torch.zeros(frames, count)
    normalized = F.normalize(appearance.float(), dim=-1)
    for frame in range(1, frames):
        if bool(shot_starts[frame]) or float(timestamps[frame] - timestamps[frame - 1]) <= 1e-6:
            continue
        left = (valid[frame - 1].bool() & (categories[frame - 1] > 0)).nonzero().flatten()
        right = (valid[frame].bool() & (categories[frame] > 0)).nonzero().flatten()
        if not len(left) or not len(right):
            continue
        cosine = normalized[frame - 1, left] @ normalized[frame, right].T
        overlap = iou_matrix(geometry[frame - 1, left, :4], geometry[frame, right, :4])
        allowed = categories[frame - 1, left, None] == categories[frame, None, right]
        scores = (0.75 * cosine + 0.25 * overlap).masked_fill(~allowed, -100.0)
        rows, columns = linear_sum_assignment((-scores).numpy())
        for row, column in zip(rows, columns):
            if not allowed[row, column] or cosine[row, column] < 0.5 or scores[row, column] < 0.45:
                continue
            other_row = scores[row].clone(); other_row[column] = -100.0
            other_column = scores[:, column].clone(); other_column[row] = -100.0
            competitor = max(float(other_row.max()), float(other_column.max()), 0.0)
            margin = max(0.0, float(scores[row, column]) - competitor)
            quality = float(scores[row, column].clamp(0, 1)) * min(1.0, margin / 0.2)
            previous[frame, right[column]] = left[row]
            reliability[frame, right[column]] = quality
    return previous, reliability


def make_events(spatial, appearance, shot_starts, maximum_events=504):
    """Build fixed-shape event candidates, independent of captions and variant.

Geometry = current relation(9), previous relation(9), rate(9), normalized
time, time squared, sin(time*pi), cos(time*pi), elapsed seconds: 32 values.
The rate follows the same ordered physical pair through the association.
"""
    times = spatial['timestamps'].float()
    valid = spatial['object_mask'].bool() & spatial['frame_mask'].bool()[:, None]
    previous, match_quality = associate(
        spatial['object_ids'], spatial['object_geometry'].float(), appearance,
        valid, times, shot_starts)
    roi_dim = appearance.shape[-1]
    result = {
        'event_appearance': torch.zeros(maximum_events, roi_dim * 2, dtype=torch.float16),
        'event_geometry': torch.zeros(maximum_events, 32),
        'event_reliability': torch.zeros(maximum_events, 5),
        'event_change': torch.zeros(maximum_events),
        'event_mask': torch.zeros(maximum_events, dtype=torch.bool),
        'event_frame': torch.full((maximum_events,), -1, dtype=torch.long),
        'association_previous': previous,
        'association_reliability': match_quality,
    }
    present_times = times[spatial['frame_mask'].bool()]
    duration = float((present_times.max() - present_times.min()).clamp_min(1e-6))
    origin = float(present_times.min())
    offset = 0
    for frame in range(1, len(times)):
        if bool(shot_starts[frame]):
            continue
        candidates = (valid[frame] & (previous[frame] >= 0)).nonzero().flatten().tolist()
        for first in candidates:
            for second in candidates:
                if first == second:
                    continue
                if offset >= maximum_events:
                    raise ValueError('Event budget smaller than number of valid pairs')
                p, q = int(previous[frame, first]), int(previous[frame, second])
                elapsed = float(times[frame] - times[frame - 1])
                current = spatial['object_relations'][frame, first, second].float()
                old = spatial['object_relations'][frame - 1, p, q].float()
                change = ((current - old) / max(elapsed, 0.04)).clamp(-5, 5)
                time = torch.tensor((float(times[frame]) - origin) / duration)
                time_features = torch.stack([time, time.square(), (time * torch.pi).sin(),
                                             (time * torch.pi).cos(), time.new_tensor(elapsed)])
                result['event_appearance'][offset] = torch.cat([
                    appearance[frame, first], appearance[frame, second]]).half()
                result['event_geometry'][offset] = torch.cat([current, old, change, time_features])
                a = spatial['object_geometry'][frame, first, 9]
                b = spatial['object_geometry'][frame, second, 9]
                result['event_reliability'][offset] = torch.tensor([
                    float(a), float(b), float(match_quality[frame, first]),
                    float(match_quality[frame, second]), min(1.0, elapsed / duration)])
                result['event_change'][offset] = torch.linalg.vector_norm(change[:3])
                result['event_mask'][offset] = True
                result['event_frame'][offset] = frame
                offset += 1
    return result
