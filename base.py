"""Detector interface. Every detector must declare whether it is a real model."""
from abc import ABC, abstractmethod

import numpy as np


class Detector(ABC):
    name: str = "unnamed"
    is_mock: bool = True   # real trained models must set False

    @abstractmethod
    def score(self, samples: np.ndarray, sample_rate: int) -> float:
        """Return synthetic-speech probability in [0, 1] for one window."""