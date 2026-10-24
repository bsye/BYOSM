from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .metrics import QUANTILES

MAX_LAGS = 96
ORIGINS_PER_SERIES = 40
MAX_ROWS = 1_500_000
CALIB_SHARE = 0.1
ITERATIONS = 300
SCALE_SEASONS = 7
SEED = 0
MIN_NOISE = 1e-3
SEASONAL_COLUMN = -3
STEP_COLUMN = -2
MAX_UNITS = 10.0


def _scale(history: np.ndarray, season: int) -> float:
    tail = history[-max(SCALE_SEASONS * season, 8) :]
    tail = tail[~np.isnan(tail)]
    value = float(np.mean(np.abs(tail))) if len(tail) else 1.0
    return value or 1.0


def _noise(history: np.ndarray, season: int, scale: float) -> float:
    tail = history[-max(SCALE_SEASONS * season, 8) - season :]
    diffs = np.abs(tail[season:] - tail[:-season]) if len(tail) > season else np.zeros(0)
    diffs = diffs[~np.isnan(diffs)]
    return max(float(diffs.mean()) / scale if len(diffs) else 1.0, MIN_NOISE)


@dataclass
class Group:
    histories: list[np.ndarray]
    season: int
    horizon: int


def _lags(season: int) -> int:
    return min(MAX_LAGS, max(2 * season, 24))


class SeasonalNaive:
    name = "seasonal naive"

    def fit(self, groups: list[Group]) -> None:
        pass

    def predict_many(self, histories: list[np.ndarray], season: int, horizon: int) -> list[np.ndarray]:
        return [self._one(h, season, horizon) for h in histories]

    def _one(self, history: np.ndarray, season: int, horizon: int) -> np.ndarray:
        s = min(season, len(history))
        point = np.array([history[-s + (h % s)] for h in range(horizon)])
        resid = history[s:] - history[:-s] if len(history) > s else np.zeros(1)
        spread = (
            np.nanquantile(resid[~np.isnan(resid)], QUANTILES) if np.isfinite(resid).any() else np.zeros(len(QUANTILES))
        )
        return np.nan_to_num(point[None] + spread[:, None])


class Boosted:
    name = "gradient boosting"

    def __init__(self, effort: int = 0) -> None:
        self.effort = effort

    def _rows(self, history: np.ndarray, origin: int, horizon: int, scale: float, season: int) -> np.ndarray:
        lags = history[max(0, origin - MAX_LAGS) : origin] / scale
        lags = np.concatenate([np.full(MAX_LAGS - len(lags), np.nan), lags])
        recent = history[max(0, origin - season) : origin] / scale
        present = recent[~np.isnan(recent)]
        level = [present.mean(), present.std(), present.min(), present.max()] if len(present) else [np.nan] * 4
        steps = np.arange(horizon)
        back = origin - season + steps % season
        seasonal = np.where((back >= 0) & (back < origin), history[np.clip(back, 0, len(history) - 1)] / scale, np.nan)
        per_step = np.column_stack([seasonal, steps, (origin + steps) % season])
        shared = np.tile(np.concatenate([lags, level, [season]]), (horizon, 1))
        return np.hstack([shared, per_step])

    def _base(self, x: np.ndarray) -> np.ndarray:
        seasonal, last = x[:, SEASONAL_COLUMN], x[:, MAX_LAGS - 1]
        return np.nan_to_num(np.where(np.isnan(seasonal), last, seasonal))

    def _table(self, groups: list[Group], per_series: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        xs, ys, late = [], [], []
        for g in groups:
            start = _lags(g.season) // 2
            for h in g.histories:
                usable = len(h) - g.horizon
                if usable <= start:
                    continue
                origins = np.unique(np.linspace(start, usable - 1, min(per_series, usable)).astype(int))
                cut = origins[int(len(origins) * (1 - CALIB_SHARE))] if len(origins) > 1 else usable
                for origin in origins:
                    scale = _scale(h[:origin], g.season)
                    rows = self._rows(h, origin, g.horizon, scale, g.season)
                    spread = _noise(h[:origin], g.season, scale)
                    target = (h[origin : origin + g.horizon] / scale - self._base(rows)) / spread
                    xs.append(rows)
                    ys.append(np.clip(target, -MAX_UNITS, MAX_UNITS))
                    late.append(np.full(g.horizon, origin >= cut))
        return np.vstack(xs), np.concatenate(ys), np.concatenate(late)

    def fit(self, groups: list[Group]) -> None:
        from sklearn.ensemble import HistGradientBoostingRegressor

        cells = sum(len(g.histories) * g.horizon for g in groups)
        x, y, late = self._table(groups, max(ORIGINS_PER_SERIES, MAX_ROWS // max(1, cells)))
        known = ~np.isnan(y)
        x, y, late = x[known], y[known], late[known]
        iterations = ITERATIONS * 2**self.effort
        model = HistGradientBoostingRegressor(loss="squared_error", max_iter=iterations, random_state=SEED)
        self.model = model.fit(x[~late], y[~late])
        steps = x[late][:, STEP_COLUMN].astype(int)
        errors = y[late] - self.model.predict(x[late])
        longest = max(g.horizon for g in groups)
        spread = [
            np.quantile(errors[steps == k], QUANTILES) if (steps == k).sum() > 2 else np.zeros(len(QUANTILES))
            for k in range(longest)
        ]
        self.spread = np.asarray(spread).T
        self.model.fit(x, y)

    def predict_many(self, histories: list[np.ndarray], season: int, horizon: int) -> list[np.ndarray]:
        scales = [_scale(h, season) for h in histories]
        x = np.vstack([self._rows(h, len(h), horizon, sc, season) for h, sc in zip(histories, scales, strict=True)])
        base = self._base(x).reshape(len(histories), horizon)
        median = self.model.predict(x).reshape(len(histories), horizon)
        noise = [_noise(h, season, sc) for h, sc in zip(histories, scales, strict=True)]
        spread = self.spread[:, :horizon]
        return [
            np.sort((base[i][None] + (median[i][None] + spread) * noise[i]) * sc, axis=0) for i, sc in enumerate(scales)
        ]


def rivals() -> list[Any]:
    out: list[Any] = [SeasonalNaive()]
    try:
        from .timestral_rival import Timestral

        out.append(Timestral())
    except ImportError:
        pass
    return out
