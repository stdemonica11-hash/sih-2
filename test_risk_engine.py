import numpy as np

from app.audio.features import analyze_window
from app.risk_engine.engine import RiskEngine

SR = 16000


def _speech_like() -> np.ndarray:
    rng = np.random.default_rng(1)
    t = np.arange(int(SR * 3.5)) / SR
    x = 0.2 * np.sin(2 * np.pi * 150 * t) + 0.05 * rng.standard_normal(t.size)
    return x.astype(np.float32)


def _features():
    return analyze_window(_speech_like(), SR)


def test_sustained_high_scores_reach_high_risk():
    eng = RiskEngine()
    out = None
    for _ in range(8):
        out = eng.update(0.9, _features(), is_mock=True)
    assert out.risk >= 61
    assert out.level in ("HIGH", "CRITICAL")


def test_sustained_low_scores_stay_low():
    eng = RiskEngine()
    out = None
    for _ in range(8):
        out = eng.update(0.05, _features(), is_mock=True)
    assert out.level == "LOW"


def test_quality_lowers_confidence_not_risk():
    eng_clean, eng_dirty = RiskEngine(), RiskEngine()
    clean = _features()
    clipped = analyze_window(np.clip(_speech_like() * 6, -1, 1), SR)
    o_clean = eng_clean.update(0.5, clean, is_mock=True)
    o_dirty = eng_dirty.update(0.5, clipped, is_mock=True)
    assert o_dirty.confidence < o_clean.confidence
    assert abs(o_dirty.risk - o_clean.risk) <= 5   # quality must not inflate risk


def test_reasons_always_present_and_mock_labeled():
    eng = RiskEngine()
    out = eng.update(0.7, _features(), is_mock=True)
    assert out.reasons and "MOCK" in out.reasons[0]