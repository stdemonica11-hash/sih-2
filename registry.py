"""Selects the active detector at startup. Falls back to mock with a loud log."""
import logging
import os

from app.config import settings
from app.inference.base import Detector
from app.inference.mock_detector import MockHeuristicDetector

log = logging.getLogger("inference")

_detector: Detector | None = None


def get_detector() -> Detector:
    global _detector
    if _detector is None:
        if settings.model_path and os.path.isfile(settings.model_path):
            try:
                from app.inference.aasist_onnx import AasistOnnxDetector
                _detector = AasistOnnxDetector(settings.model_path)
                log.info("Loaded REAL model: %s", settings.model_path)
            except Exception:
                log.exception("Failed to load ONNX model; falling back to MOCK detector")
                _detector = MockHeuristicDetector()
        else:
            log.warning("No model configured (VID_MODEL_PATH empty) -> MOCK detector active")
            _detector = MockHeuristicDetector()
    return _detector