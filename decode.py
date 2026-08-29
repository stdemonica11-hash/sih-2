"""Safe audio decoding. Treats every upload as untrusted input."""
import io
import numpy as np
import soundfile as sf

from app.config import settings


class AudioDecodeError(Exception):
    pass


def decode_bytes(data: bytes) -> tuple[np.ndarray, int]:
    """Decode audio bytes -> (mono float32 array, sample_rate).

    soundfile/libsndfile sniffs the real container format from magic bytes,
    so a mislabeled or malicious extension cannot select a decoder.
    """
    if len(data) > settings.max_upload_mb * 1024 * 1024:
        raise AudioDecodeError(f"File exceeds {settings.max_upload_mb} MB limit")
    try:
        audio, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
    except Exception as exc:  # noqa: BLE001 - anything from libsndfile = bad input
        raise AudioDecodeError(f"Unsupported or corrupt audio file: {exc}") from exc

    duration = audio.shape[0] / sr
    if duration > settings.max_duration_sec:
        raise AudioDecodeError(
            f"Audio too long ({duration:.0f}s > {settings.max_duration_sec:.0f}s limit)"
        )
    if audio.shape[0] == 0:
        raise AudioDecodeError("Audio file contains no samples")

    mono = audio.mean(axis=1)  # downmix
    return mono, sr