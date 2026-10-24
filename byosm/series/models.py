from __future__ import annotations

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


def _scale(history: np.ndarray, season: int) -> float:
    tail = history[-max(SCALE_SEASONS * season, 8) :]
    tail = tail[~np.isnan(tail)]
    value = float(np.mean(np.abs(tail))) if len(tail) else 1.0
    return value or 1.0


class SeasonalNaive:
    name = "seasonal naive"

    def fit(self, histories: list[np.ndarray], season: int, horizon: int) -> None:
        self.season = season

    def predict(self, history: np.ndarray, horizon: int) -> np.ndarray:
        s = min(self.season, len(history))
        point = np.array([history[-s + (h % s)] for h in range(horizon)])
        resid = history[s:] - history[:-s] if len(history) > s else np.zeros(1)
        spread = (
            np.nanquantile(resid[~np.isnan(resid)], QUANTILES) if np.isfinite(resid).any() else np.zeros(len(QUANTILES))
        )
        return np.nan_to_num(point[None] + spread[:, None])


class Boosted:
    name = "gradient boosting"

    def _rows(self, history: np.ndarray, origin: int, horizon: int, scale: float) -> np.ndarray:
        lags = history[max(0, origin - self.lags) : origin] / scale
        lags = np.concatenate([np.full(self.lags - len(lags), np.nan), lags])
        recent = history[max(0, origin - self.season) : origin] / scale
        present = recent[~np.isnan(recent)]
        level = [present.mean(), present.std(), present.min(), present.max()] if len(present) else [np.nan] * 4
        steps = np.arange(horizon)
        back = origin - self.season + steps % self.season
        seasonal = np.where((back >= 0) & (back < origin), history[np.clip(back, 0, len(history) - 1)] / scale, np.nan)
        per_step = np.column_stack([seasonal, steps, (origin + steps) % self.season])
        shared = np.tile(np.concatenate([lags, level]), (horizon, 1))
        return np.hstack([shared, per_step])

    def _table(
        self, histories: list[np.ndarray], horizon: int, per_series: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        xs, ys, late = [], [], []
        for h in histories:
            usable = len(h) - horizon
            if usable <= self.lags // 2:
                continue
            origins = np.unique(np.linspace(self.lags // 2, usable - 1, min(per_series, usable)).astype(int))
            cut = origins[int(len(origins) * (1 - CALIB_SHARE))] if len(origins) > 1 else usable
            for origin in origins:
                scale = _scale(h[:origin], self.season)
                xs.append(self._rows(h, origin, horizon, scale))
                ys.append(h[origin : origin + horizon] / scale)
                late.append(np.full(horizon, origin >= cut))
        return np.vstack(xs), np.concatenate(ys), np.concatenate(late)

    def fit(self, histories: list[np.ndarray], season: int, horizon: int) -> None:
        from sklearn.ensemble import HistGradientBoostingRegressor

        self.season, self.lags, self.horizon = season, min(MAX_LAGS, max(2 * season, 24)), horizon
        per_series = max(ORIGINS_PER_SERIES, MAX_ROWS // max(1, len(histories) * horizon))
        x, y, late = self._table(histories, horizon, per_series)
        known = ~np.isnan(y)
        x, y, late = x[known], y[known], late[known]
        model = HistGradientBoostingRegressor(loss="squared_error", max_iter=ITERATIONS, random_state=SEED)
        self.model = model.fit(x[~late], y[~late])
        steps = x[late][:, -2].astype(int)
        errors = y[late] - self.model.predict(x[late])
        spread = [
            np.quantile(errors[steps == k], QUANTILES) if (steps == k).sum() > 2 else np.zeros(len(QUANTILES))
            for k in range(horizon)
        ]
        self.spread = np.asarray(spread).T
        self.model.fit(x, y)

    def predict(self, history: np.ndarray, horizon: int) -> np.ndarray:
        return self.predict_many([history], horizon)[0]

    def predict_many(self, histories: list[np.ndarray], horizon: int) -> list[np.ndarray]:
        scales = [_scale(h, self.season) for h in histories]
        x = np.vstack([self._rows(h, len(h), horizon, sc) for h, sc in zip(histories, scales, strict=True)])
        median = self.model.predict(x).reshape(len(histories), horizon)
        return [np.sort((median[i][None] + self.spread) * sc, axis=0) for i, sc in enumerate(scales)]


def rivals() -> list[Any]:
    out: list[Any] = [SeasonalNaive()]
    try:
        from .timestral_rival import Timestral

        out.append(Timestral())
    except ImportError:
        pass
    return out
