import io

import numpy as np
import soundfile as sf
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def _wav_bytes(sec: float = 6.0, sr: int = 16000) -> bytes:
    rng = np.random.default_rng(2)
    t = np.arange(int(sr * sec)) / sr
    x = 0.25 * np.sin(2 * np.pi * 180 * t) + 0.04 * rng.standard_normal(t.size)
    buf = io.BytesIO()
    sf.write(buf, x.astype(np.float32), sr, format="WAV")
    return buf.getvalue()


def test_health_reports_mock_model():
    r = client.get("/api/v1/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["model"]["is_mock"] is True
    assert "window_sec" in body["config"]


def test_score_endpoint_full_pipeline():
    r = client.post("/api/v1/score", files={"file": ("test.wav", _wav_bytes(), "audio/wav")})
    assert r.status_code == 200
    body = r.json()
    assert body["n_windows"] > 0
    assert body["n_scored"] > 0
    assert body["final_risk"] is not None
    assert 0 <= body["final_risk"]["risk"] <= 100
    assert body["model"]["is_mock"] is True
    assert body["disclaimer"]
    first_scored = next(w for w in body["windows"] if w["speech"])
    assert 0.0 <= first_scored["score"] <= 1.0
    assert "flatness" in first_scored["features"]


def test_silent_file_scores_no_windows():
    buf = io.BytesIO()
    sf.write(buf, np.zeros(16000 * 5, dtype=np.float32), 16000, format="WAV")
    r = client.post("/api/v1/score", files={"file": ("silent.wav", buf.getvalue(), "audio/wav")})
    assert r.status_code == 200
    body = r.json()
    assert body["n_scored"] == 0
    assert body["final_risk"] is None


def test_garbage_file_rejected_422():
    r = client.post("/api/v1/score", files={"file": ("evil.wav", b"not audio at all", "audio/wav")})
    assert r.status_code == 422