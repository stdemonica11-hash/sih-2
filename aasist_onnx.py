"""AASIST ONNX detector adapter (Phase 6 target).

Loads a real exported AASIST model when VID_MODEL_PATH points at a .onnx
file. Expected model contract (standard AASIST export):
  input : float32 [batch, 64600]  raw waveform @ 16 kHz
  output: float32 [batch, 2]      logits [bonafide, spoof]
"""
import numpy as np

from app.config import settings
from app.inference.base import Detector


class AasistOnnxDetector(Detector):
    name = "aasist-onnx"
    is_mock = False

    def __init__(self, model_path: str):
        import onnxruntime as ort  # imported lazily; only needed with a real model
        self.session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.n_in = settings.model_input_samples

    def _fit_length(self, x: np.ndarray) -> np.ndarray:
        if x.size >= self.n_in:
            return x[: self.n_in]
        reps = int(np.ceil(self.n_in / x.size))
        return np.tile(x, reps)[: self.n_in]

    def score(self, samples: np.ndarray, sample_rate: int) -> float:
        if sample_rate != settings.target_sample_rate:
            raise ValueError("AASIST expects 16 kHz input; resample upstream")
        x = self._fit_length(samples.astype(np.float32))[None, :]
        logits = self.session.run(None, {self.input_name: x})[0][0]
        exp = np.exp(logits - logits.max())
        probs = exp / exp.sum()
        return float(probs[1])   # index 1 = spoof class in standard AASIST training