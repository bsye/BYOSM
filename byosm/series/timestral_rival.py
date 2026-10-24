from __future__ import annotations

import numpy as np
from timestral import TimestralForecaster

from .metrics import QUANTILES

REPO = "mistralai/Timestral-209M"


class Timestral:
    name = "Timestral-209M"

    def fit(self, histories: list[np.ndarray], season: int, horizon: int) -> None:
        self.model = TimestralForecaster.from_pretrained(REPO, device="cpu")

    def predict(self, history: np.ndarray, horizon: int) -> np.ndarray:
        forecast = self.model.forecast(history, prediction_length=horizon)
        return np.stack([np.asarray(forecast.quantile(q), dtype=float) for q in QUANTILES])
