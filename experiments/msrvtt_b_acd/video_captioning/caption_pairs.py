"""Enumerate every annotated training pair while reusing frozen video features."""
from functools import lru_cache

from .data import CachedVideoDataset


class AllCaptionDataset(CachedVideoDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.split != 'train':
            raise ValueError('All-caption enumeration is restricted to training')
        self.pairs = [(video, caption) for video, row in enumerate(self.rows)
                      for caption in range(len(row['references']))]

    def __len__(self):
        return len(self.pairs)

    @lru_cache(maxsize=128)
    def video_features(self, video_index):
        return super().__getitem__(video_index)[0]

    def __getitem__(self, index):
        video, caption = self.pairs[index]
        return self.video_features(video), self.rows[video]['references'][caption]
