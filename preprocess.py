"""Resampling and normalization."""
import numpy as np
from math import gcd
from scipy.signal import resample_poly

from app.config import settings


def resample(mono: np.ndarray, sr: int, target_sr: int | None = None) -> tuple[np.ndarray, int]:
    target_sr = target_sr or settings.target_sample_rate
    if sr == target_sr:
        return mono.astype(np.float32), sr
    g = gcd(sr, target_sr)
    out = resample_poly(mono, target_sr // g, sr // g).astype(np.float32)
    return out, target_sr


def peak_normalize(mono: np.ndarray, peak_target: float = 0.95) -> np.ndarray:
    peak = float(np.max(np.abs(mono))) if mono.size else 0.0
    if peak < 1e-6:
        return mono
    return (mono * (peak_target / peak)).astype(np.float32) if peak > peak_target else mono