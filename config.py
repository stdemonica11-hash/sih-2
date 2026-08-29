"""Central configuration. All values overridable via environment / .env file."""
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # --- audio pipeline ---
    target_sample_rate: int = 16000
    window_sec: float = 3.5
    hop_sec: float = 1.0
    vad_rms_threshold: float = 0.012     # energy gate: below this a window is "no speech"
    max_upload_mb: int = 50
    max_duration_sec: float = 600.0

    # --- risk engine (operational defaults, NOT scientific constants) ---
    synth_threshold: float = 0.5
    w_synth: float = 0.65
    w_streak: float = 0.25
    w_instability: float = 0.10
    ema_alpha: float = 0.35
    risk_bands: tuple[int, int, int] = (30, 60, 80)   # LOW/MED/HIGH/CRIT boundaries

    # --- model ---
    model_path: str = ""                 # path to AASIST .onnx; empty -> mock detector
    model_input_samples: int = 64600     # AASIST standard input length (~4.04 s @ 16 kHz)

    # --- server ---
    cors_origins: list[str] = ["*"]      # tighten for production
    rate_limit_per_minute: int = 60

    class Config:
        env_file = ".env"
        env_prefix = "VID_"              # e.g. VID_MODEL_PATH=/models/aasist.onnx


settings = Settings()