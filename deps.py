"""Shared helpers used by both REST and WebSocket routes."""
import numpy as np

from app.audio import vad
from app.audio.features import analyze_window
from app.inference.base import Detector
from app.risk_engine.engine import RiskEngine

MOCK_DISCLAIMER = (
    "Scores from mock-heuristic detectors are placeholders and are NOT "
    "validated synthetic-speech detection results."
)


def model_info_dict(det: Detector) -> dict:
    return {
        "name": det.name,
        "is_mock": det.is_mock,
        "note": MOCK_DISCLAIMER if det.is_mock else
                "Trained anti-spoofing model; see docs/robustness_report.md for evaluated performance.",
    }


def score_one_window(
    det: Detector, engine: RiskEngine, samples: np.ndarray, sr: int, t_sec: float
) -> dict:
    features = analyze_window(samples, sr)
    if features is None:
        return {"t_sec": round(t_sec, 2), "speech": False, "score": None,
                "features": {}, "risk": None}
    if not vad.has_speech(features):
        return {"t_sec": round(t_sec, 2), "speech": False, "score": None,
                "features": features.to_dict(), "risk": None}
    score = det.score(samples, sr)
    risk = engine.update(score, features, det.is_mock)
    return {"t_sec": round(t_sec, 2), "speech": True, "score": round(score, 4),
            "features": features.to_dict(), "risk": risk.to_dict()}