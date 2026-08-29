"""Forensic scoring endpoint: upload a file, get the full window-by-window analysis."""
import time

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from app.api.deps import MOCK_DISCLAIMER, model_info_dict, score_one_window
from app.api.schemas import ScoreResponse
from app.audio.decode import AudioDecodeError, decode_bytes
from app.audio.preprocess import peak_normalize, resample
from app.audio.windowing import iter_windows
from app.config import settings
from app.inference.registry import get_detector
from app.risk_engine.engine import RiskEngine
from app.utils.security import rate_limit

router = APIRouter()


@router.post("/score", response_model=ScoreResponse, dependencies=[Depends(rate_limit)])
async def score_file(file: UploadFile = File(...)) -> dict:
    t0 = time.perf_counter()
    data = await file.read(settings.max_upload_mb * 1024 * 1024 + 1)
    try:
        mono, sr = decode_bytes(data)
    except AudioDecodeError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    mono, sr = resample(mono, sr)
    mono = peak_normalize(mono)

    det = get_detector()
    engine = RiskEngine()
    windows = [score_one_window(det, engine, seg, sr, t) for t, seg in iter_windows(mono, sr)]
    scored = [w for w in windows if w["speech"]]
    final_risk = scored[-1]["risk"] if scored else None

    return {
        "model": model_info_dict(det),
        "sample_rate": sr,
        "duration_sec": round(mono.size / sr, 2),
        "n_windows": len(windows),
        "n_scored": len(scored),
        "windows": windows,
        "final_risk": final_risk,
        "processing_ms": round((time.perf_counter() - t0) * 1000, 1),
        "disclaimer": MOCK_DISCLAIMER if det.is_mock else "",
    }