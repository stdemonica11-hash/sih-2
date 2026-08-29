"""Risk engine: fuses detector output with temporal and quality evidence.

Design principles (from project spec):
- Model output is evidence, not the final decision.
- Poor audio quality LOWERS confidence; it never raises risk.
- Every reason string maps 1:1 to a computed quantity.
- Weights/bands are operational defaults, not scientific constants.
"""
from dataclasses import dataclass, field

import numpy as np

from app.audio.features import WindowFeatures
from app.config import settings


@dataclass
class RiskOutput:
    risk: int
    level: str
    ema: float
    streak: float
    instability: float
    confidence: float
    reasons: list[str]

    def to_dict(self) -> dict:
        return {
            "risk": self.risk, "level": self.level,
            "ema": round(self.ema, 4), "streak": round(self.streak, 4),
            "instability": round(self.instability, 4),
            "confidence": round(self.confidence, 4),
            "reasons": self.reasons,
        }


@dataclass
class RiskEngine:
    ema: float = 0.0
    initialized: bool = False
    recent: list[float] = field(default_factory=list)

    def reset(self) -> None:
        self.ema, self.initialized = 0.0, False
        self.recent.clear()

    def update(self, score: float, features: WindowFeatures, is_mock: bool) -> RiskOutput:
        a = settings.ema_alpha
        self.ema = a * score + (1 - a) * self.ema if self.initialized else score
        self.initialized = True
        self.recent.append(score)
        if len(self.recent) > 8:
            self.recent.pop(0)

        n = len(self.recent)
        streak = sum(s > settings.synth_threshold for s in self.recent) / n
        instability = min(1.0, float(np.std(self.recent)) / 0.35)

        confidence, quality_flags = 1.0, []
        if features.clip_frac > 0.01:
            confidence -= 0.2
            quality_flags.append(f"Clipping in {features.clip_frac * 100:.1f}% of samples")
        if features.rms < settings.vad_rms_threshold * 1.5:
            confidence -= 0.25
            quality_flags.append("Low speech energy - reliability reduced")
        if features.rolloff_hz < 3500:
            confidence -= 0.15
            quality_flags.append(
                f"Narrowband audio (rolloff {features.rolloff_hz / 1000:.1f} kHz) - possible phone codec"
            )
        confidence = max(0.2, confidence)

        fused = (settings.w_synth * self.ema
                 + settings.w_streak * streak
                 + settings.w_instability * instability)
        risk = int(round(100 * float(np.clip(fused, 0, 1))))
        b1, b2, b3 = settings.risk_bands
        level = "LOW" if risk <= b1 else "MEDIUM" if risk <= b2 else "HIGH" if risk <= b3 else "CRITICAL"

        reasons = [f"Smoothed detector score {self.ema * 100:.0f}%"
                   + (" (MOCK model)" if is_mock else "")]
        if streak > 0.5:
            reasons.append(
                f"{round(streak * n)} of last {n} windows above "
                f"{settings.synth_threshold * 100:.0f}% threshold"
            )
        if features.flatness > 0.3:
            reasons.append(f"Elevated spectral flatness ({features.flatness:.2f})")
        if features.high_ratio < 0.05:
            reasons.append(f"Only {features.high_ratio * 100:.1f}% energy above 4 kHz")
        if instability > 0.6:
            reasons.append("Detector score unstable across windows")
        reasons.extend(quality_flags)

        return RiskOutput(risk, level, self.ema, streak, instability, confidence, reasons)