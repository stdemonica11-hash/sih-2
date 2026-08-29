"""MOCK DETECTOR -- placeholder heuristic, NOT a trained model.

Deterministic function of real DSP features so the pipeline is fully
exercisable end-to-end, but its output is NOT a validated synthetic-speech
probability. Replaced by AasistOnnxDetector when VID_MODEL_PATH is set.
Every API response produced with this detector carries model.is_mock=true.
"""
import numpy as np

from app.audio.features import analyze_window
from app.inference.base import Detector


class MockHeuristicDetector(Detector):
    name = "mock-heuristic-v0"
    is_mock = True

    def score(self, samples: np.ndarray, sample_rate: int) -> float:
        f = analyze_window(samples, sample_rate)
        if f is None:
            return 0.5
        flat = min(1.0, f.flatness / 0.5)
        narrow = 1.0 - min(1.0, f.high_ratio / 0.10)
        zcr_odd = min(1.0, abs(f.zcr - 0.07) / 0.12)
        raw = 0.45 * flat + 0.35 * narrow + 0.20 * zcr_odd
        return float(np.clip(raw, 0.02, 0.98))