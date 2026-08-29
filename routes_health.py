from fastapi import APIRouter

from app.api.deps import model_info_dict
from app.api.schemas import HealthResponse
from app.config import settings
from app.inference.registry import get_detector

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health() -> dict:
    det = get_detector()
    return {
        "status": "ok",
        "model": model_info_dict(det),
        "config": {
            "window_sec": settings.window_sec,
            "hop_sec": settings.hop_sec,
            "target_sample_rate": settings.target_sample_rate,
            "risk_bands": list(settings.risk_bands),
            "synth_threshold": settings.synth_threshold,
        },
    }