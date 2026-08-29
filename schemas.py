"""Pydantic response models = the API contract the frontend codes against."""
from pydantic import BaseModel


class ModelInfo(BaseModel):
    name: str
    is_mock: bool
    note: str


class WindowResult(BaseModel):
    t_sec: float
    speech: bool
    score: float | None          # synthetic probability for this window
    features: dict
    risk: dict | None            # risk engine output after this window


class ScoreResponse(BaseModel):
    model: ModelInfo
    sample_rate: int
    duration_sec: float
    n_windows: int
    n_scored: int
    windows: list[WindowResult]
    final_risk: dict | None
    processing_ms: float
    disclaimer: str


class HealthResponse(BaseModel):
    status: str
    model: ModelInfo
    config: dict