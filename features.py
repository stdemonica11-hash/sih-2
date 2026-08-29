"""Real DSP feature extraction for one audio window (numpy only)."""
from dataclasses import dataclass, asdict

import numpy as np

FFT_N = 2048


@dataclass
class WindowFeatures:
    rms: float
    zcr: float
    clip_frac: float
    peak: float
    centroid_hz: float
    rolloff_hz: float
    flatness: float
    high_ratio: float          # fraction of spectral energy above 4 kHz
    sample_rate: int

    def to_dict(self) -> dict:
        return {k: (round(v, 6) if isinstance(v, float) else v) for k, v in asdict(self).items()}


def _avg_spectrum(samples: np.ndarray) -> np.ndarray | None:
    """Average magnitude spectrum over Hann-windowed frames (hop = N/2)."""
    if samples.size < FFT_N:
        return None
    hann = np.hanning(FFT_N)
    hop = FFT_N // 2
    n_frames = 1 + (samples.size - FFT_N) // hop
    idx = np.arange(FFT_N)[None, :] + hop * np.arange(n_frames)[:, None]
    frames = samples[idx] * hann[None, :]
    mags = np.abs(np.fft.rfft(frames, axis=1))[:, : FFT_N // 2]
    return mags.mean(axis=0)


def analyze_window(samples: np.ndarray, sample_rate: int) -> WindowFeatures | None:
    if samples.size < FFT_N:
        return None
    abs_s = np.abs(samples)
    rms = float(np.sqrt(np.mean(samples**2)))
    peak = float(abs_s.max())
    clip_frac = float(np.mean(abs_s > 0.985))
    zcr = float(np.mean(np.abs(np.diff(np.signbit(samples).astype(np.int8)))))

    spec = _avg_spectrum(samples)
    if spec is None:
        return None
    spec = spec[1:]  # drop DC
    freqs = np.arange(1, FFT_N // 2) * (sample_rate / FFT_N)

    spec_sum = float(spec.sum())
    centroid = float((spec * freqs).sum() / spec_sum) if spec_sum > 0 else 0.0
    energy = spec**2
    tot_e = float(energy.sum())
    high_ratio = float(energy[freqs >= 4000].sum() / tot_e) if tot_e > 0 else 0.0
    flatness = float(np.exp(np.mean(np.log(spec + 1e-12))) / (np.mean(spec) + 1e-12))

    cum = np.cumsum(spec)
    roll_idx = int(np.searchsorted(cum, 0.85 * spec_sum))
    rolloff = float(freqs[min(roll_idx, freqs.size - 1)])

    return WindowFeatures(
        rms=rms, zcr=zcr, clip_frac=clip_frac, peak=peak,
        centroid_hz=centroid, rolloff_hz=rolloff,
        flatness=flatness, high_ratio=high_ratio, sample_rate=sample_rate,
    )