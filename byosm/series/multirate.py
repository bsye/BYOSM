from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import numpy as np

from .metrics import QUANTILES, pinball

BACKTESTS = 2
SEASONS_NEEDED = 2
COARSE_CONTEXT = 1024
MIN_LOSS = 1e-9
SHARPNESS = 10.0
SUFFIX = ", multi-rate"


def twin_name(name: str) -> str:
    return name + SUFFIX


def coarse(history: np.ndarray, rate: int) -> np.ndarray:
    if rate == 1:
        return history
    blocks = history[len(history) % rate :][-COARSE_CONTEXT * rate :].reshape(-1, rate)
    seen = ~np.isnan(blocks)
    counts = seen.sum(1)
    return np.where(counts > 0, np.where(seen, blocks, 0.0).sum(1) / np.maximum(counts, 1), np.nan)


def coarse_season(season: int, rate: int) -> int:
    return max(1, round(season / rate))


def native(quantiles: np.ndarray, rate: int, horizon: int) -> np.ndarray:
    if rate == 1:
        return quantiles[:, :horizon]
    centres = np.arange(quantiles.shape[1]) * rate + (rate - 1) / 2
    return np.stack([np.interp(np.arange(horizon), centres, q) for q in quantiles])


def _predict(fn: Callable[..., list[np.ndarray]], histories: list[np.ndarray], *args: Any) -> list[np.ndarray]:
    return fn(histories, *args) if histories else []


def _part(items: list[Any], usable: list[bool], keep: bool) -> list[Any]:
    return [x for x, u in zip(items, usable, strict=True) if u == keep]


class MultiRate:
    def __init__(self, model: Any, rates: list[int]) -> None:
        self.model, self.rates, self.name = model, list(rates), twin_name(model.name)

    def _at_rates(self, histories: list[np.ndarray], season: int, horizon: int, clocks: list) -> np.ndarray:
        out = []
        for rate in self.rates:
            shape = coarse_season(season, rate), math.ceil(horizon / rate)
            pred = self.model.predict_many([coarse(h, rate) for h in histories], *shape, clocks if rate == 1 else None)
            out.append([native(q, rate, horizon) for q in pred])
        return np.asarray(out).swapaxes(0, 1)

    def _weights(self, histories: list[np.ndarray], season: int, horizon: int, clocks: list) -> np.ndarray:
        loss = np.zeros((len(histories), len(self.rates), len(QUANTILES)))
        for back in range(1, BACKTESTS + 1):
            cuts = [len(h) - back * horizon for h in histories]
            earlier = [h[:c] for h, c in zip(histories, cuts, strict=True)]
            forecasts = self._at_rates(earlier, season, horizon, clocks)
            for i, (h, c) in enumerate(zip(histories, cuts, strict=True)):
                loss[i] += [pinball(h[c : c + horizon], f) for f in forecasts[i]]
        relative = loss / np.maximum(loss.min(1, keepdims=True), MIN_LOSS)
        weight = np.exp(-SHARPNESS * (relative - 1))
        return weight / weight.sum(1, keepdims=True)

    def _mix(self, histories: list[np.ndarray], season: int, horizon: int, clocks: list) -> list[np.ndarray]:
        weights = self._weights(histories, season, horizon, clocks)
        forecasts = self._at_rates(histories, season, horizon, clocks)
        return list(np.sort(np.einsum("srq,srqh->sqh", weights, forecasts), axis=1))

    def _usable(self, history: np.ndarray, season: int, horizon: int) -> bool:
        rate = max(self.rates)
        return (len(history) - BACKTESTS * horizon) // rate > SEASONS_NEEDED * coarse_season(season, rate)

    def predict_many(self, histories: list[np.ndarray], season: int, horizon: int, clocks: Any = None) -> list:
        at, usable = clocks or [None] * len(histories), [self._usable(h, season, horizon) for h in histories]
        mixed = iter(_predict(self._mix, _part(histories, usable, True), season, horizon, _part(at, usable, True)))
        plain = _part(histories, usable, False)
        native_only = iter(_predict(self.model.predict_many, plain, season, horizon, _part(at, usable, False)))
        return [next(mixed) if u else next(native_only) for u in usable]
