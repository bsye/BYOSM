from __future__ import annotations

from typing import Any

import numpy as np

from .metrics import QUANTILES

MAX_LAGS = 96
ORIGINS_PER_SERIES = 40
MAX_ROWS = 300_000
SEED = 0


def _scale(history: np.ndarray, season: int) -> float:
    tail = history[-max(season, 8) :]
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

    def _features(self, history: np.ndarray, origin: int, step: int, scale: float) -> np.ndarray:
        lags = history[max(0, origin - self.lags) : origin] / scale
        lags = np.concatenate([np.full(self.lags - len(lags), np.nan), lags])
        back = origin - self.season + (step % self.season)
        seasonal = history[back] / scale if 0 <= back < origin else np.nan
        recent = history[max(0, origin - self.season) : origin] / scale
        level = (
            [np.nanmean(recent), np.nanstd(recent), np.nanmin(recent), np.nanmax(recent)]
            if len(recent)
            else [np.nan] * 4
        )
        return np.concatenate([lags, [seasonal, step, (origin + step) % self.season], level])

    def fit(self, histories: list[np.ndarray], season: int, horizon: int) -> None:
        from sklearn.ensemble import HistGradientBoostingRegressor

        self.season, self.lags = season, min(MAX_LAGS, max(2 * season, 24))
        rng, xs, ys = np.random.default_rng(SEED), [], []
        per_series = max(ORIGINS_PER_SERIES, MAX_ROWS // max(1, len(histories) * horizon))
        for h in histories:
            usable = len(h) - horizon
            if usable <= self.lags // 2:
                continue
            for origin in np.unique(np.linspace(self.lags // 2, usable - 1, min(per_series, usable)).astype(int)):
                scale = _scale(h[:origin], season)
                for step in range(horizon):
                    if not np.isnan(h[origin + step]):
                        xs.append(self._features(h, origin, step, scale))
                        ys.append(h[origin + step] / scale)
        x, y = np.asarray(xs), np.asarray(ys)
        if len(y) > MAX_ROWS:
            keep = rng.choice(len(y), MAX_ROWS, replace=False)
            x, y = x[keep], y[keep]
        self.models = [
            HistGradientBoostingRegressor(loss="quantile", quantile=q, max_iter=200, random_state=SEED).fit(x, y)
            for q in QUANTILES
        ]

    def predict(self, history: np.ndarray, horizon: int) -> np.ndarray:
        scale = _scale(history, self.season)
        x = np.stack([self._features(history, len(history), step, scale) for step in range(horizon)])
        return np.sort(np.stack([m.predict(x) for m in self.models]) * scale, axis=0)


def rivals() -> list[Any]:
    out: list[Any] = [SeasonalNaive()]
    try:
        from .timestral_rival import Timestral

        out.append(Timestral())
    except ImportError:
        pass
    return out
