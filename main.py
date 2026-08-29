"""SIH26104 - Voice Impersonation Detection backend.

Run:  uvicorn app.main:app --reload --port 8000
Dashboard: http://localhost:8000   ·   API docs: http://localhost:8000/docs
"""
import logging
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from app.api import routes_health, routes_score, ws_stream
from app.config import settings
from app.inference.registry import get_detector

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

app = FastAPI(
    title="SIH26104 Voice Impersonation Detection API",
    version="0.1.0",
    description="Anti-spoofing scoring pipeline. If model.is_mock is true, "
                "scores come from a placeholder heuristic, not a trained model.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(routes_health.router, prefix="/api/v1", tags=["system"])
app.include_router(routes_score.router, prefix="/api/v1", tags=["scoring"])
app.include_router(ws_stream.router, prefix="/api/v1", tags=["streaming"])


@app.on_event("startup")
def _load_model() -> None:
    get_detector()   # fail fast + log which detector is active


# ---- serve the integrated frontend (single-file dashboard) ----
_FRONTEND = Path(__file__).resolve().parent.parent / "frontend" / "index.html"


@app.get("/", include_in_schema=False)
def dashboard() -> FileResponse:
    return FileResponse(_FRONTEND, media_type="text/html")