from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np

from .metrics import summary
from .scoring import Forecasts, Prepared
from .specialist import ROUTED

FOLDS = 2


def _mean(values: np.ndarray) -> float:
    return float(np.nanmean(values)) if np.isfinite(values).any() else np.inf


def route(p: Prepared, made: dict[str, tuple[Forecasts, Forecasts]]) -> tuple[str, Forecasts, Forecasts]:
    seasons = [p.season] * len(p.calib)
    per_window = {n: np.asarray(summary(p.calib, held, seasons)["windows"]) for n, (held, _) in made.items()}
    every = np.arange(len(p.calib))

    def best(rows: np.ndarray) -> str:
        return min(per_window, key=lambda n: _mean(per_window[n][rows]))

    picks = [best(every[every % FOLDS != fold] if len(every) > 1 else every) for fold in range(FOLDS)]
    crossed = [made[picks[i % FOLDS]][0][i] for i in every]
    chosen = best(every)
    return chosen, crossed, made[chosen][1]


class Routed:
    name = ROUTED

    def __init__(self, models: dict[str, Any], routes: dict[str, str]) -> None:
        self.models, self.routes = models, routes
        self.default = Counter(routes.values()).most_common(1)[0][0]

    def for_config(self, config: str | None) -> Any:
        return self.models[self.routes.get(config or "", self.default)]

    def predict_many(self, histories: list[np.ndarray], season: int, horizon: int) -> list[np.ndarray]:
        return self.for_config(None).predict_many(histories, season, horizon)


def served(model: Any, config: str | None) -> Any:
    return model.for_config(config) if isinstance(model, Routed) else model
