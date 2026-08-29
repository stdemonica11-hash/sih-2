"""Voice activity gate.

v0: simple energy gate (honest and predictable). The interface is kept so
Silero VAD can be dropped in later without touching callers.
"""
from app.audio.features import WindowFeatures
from app.config import settings


def has_speech(features: WindowFeatures) -> bool:
    return features.rms >= settings.vad_rms_threshold