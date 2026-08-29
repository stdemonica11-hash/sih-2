"""Sliding-window segmentation over a full recording."""
from collections.abc import Iterator

import numpy as np

from app.config import settings


def iter_windows(mono: np.ndarray, sr: int) -> Iterator[tuple[float, np.ndarray]]:
    """Yield (end_time_sec, window_samples) with configured window/hop."""
    win = int(settings.window_sec * sr)
    hop = int(settings.hop_sec * sr)
    if mono.size < win:
        if mono.size >= int(1.0 * sr):     # short clip: score what we have, once
            yield mono.size / sr, mono
        return
    for start in range(0, mono.size - win + 1, hop):
        yield (start + win) / sr, mono[start : start + win]