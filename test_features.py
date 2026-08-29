"""DSP sanity checks on synthesized signals with known properties."""
import numpy as np

from app.audio.features import analyze_window

SR = 16000


def _sine(freq: float, sec: float = 3.5, amp: float = 0.5) -> np.ndarray:
    t = np.arange(int(SR * sec)) / SR
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def test_sine_has_low_flatness_and_correct_centroid():
    f = analyze_window(_sine(440), SR)
    assert f is not None
    assert f.flatness < 0.05                  # pure tone = very peaky spectrum
    assert abs(f.centroid_hz - 440) < 100     # centroid near the tone
    assert f.high_ratio < 0.01                # no energy above 4 kHz


def test_white_noise_has_high_flatness_and_high_band_energy():
    rng = np.random.default_rng(0)
    noise = (0.3 * rng.standard_normal(int(SR * 3.5))).astype(np.float32)
    f = analyze_window(noise, SR)
    assert f is not None
    assert f.flatness > 0.5                   # white noise = flat spectrum
    assert f.high_ratio > 0.3                 # plenty of energy above 4 kHz


def test_silence_measures_near_zero_rms():
    f = analyze_window(np.zeros(int(SR * 3.5), dtype=np.float32), SR)
    assert f is not None
    assert f.rms < 1e-6


def test_clipping_detected():
    x = _sine(200, amp=2.0)
    np.clip(x, -1, 1, out=x)
    f = analyze_window(x, SR)
    assert f.clip_frac > 0.1


def test_too_short_window_returns_none():
    assert analyze_window(np.zeros(100, dtype=np.float32), SR) is None