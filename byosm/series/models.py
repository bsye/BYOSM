from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .calendar import Clock
from .metrics import QUANTILES
from .rows import STEP_COLUMN, forecast_rows, table

ITERATIONS = 300
SEED = 0
BOOSTED_FOLDER = "boosted"
Clocks = list[Clock | None] | None


def average(forecasts: list[list[np.ndarray]]) -> list[np.ndarray]:
    return [np.mean(group, axis=0) for group in zip(*forecasts, strict=True)]


class Ensemble:
    name = "ensemble"

    def __init__(self, models: list[Any], name: str = name) -> None:
        self.models, self.name = models, name

    def predict_many(self, histories: list[np.ndarray], season: int, horizon: int, clocks: Clocks = None) -> list:
        return average([m.predict_many(histories, season, horizon, clocks) for m in self.models])


@dataclass
class Group:
    histories: list[np.ndarray]
    season: int
    horizon: int
    clocks: list[Clock | None] = field(default_factory=list)


class SeasonalNaive:
    name = "seasonal naive"

    def fit(self, groups: list[Group]) -> None:
        pass

    def predict_many(self, histories: list[np.ndarray], season: int, horizon: int, clocks: Clocks = None) -> list:
        return [self._one(h, season, horizon) for h in histories]

    def _one(self, history: np.ndarray, season: int, horizon: int) -> np.ndarray:
        s = min(season, len(history))
        point = np.array([history[-s + (h % s)] for h in range(horizon)])
        resid = history[s:] - history[:-s] if len(history) > s else np.zeros(1)
        spread = (
            np.nanquantile(resid[~np.isnan(resid)], QUANTILES) if np.isfinite(resid).any() else np.zeros(len(QUANTILES))
        )
        return np.nan_to_num(point[None] + spread[:, None])


def _boosted_path(name: str) -> Path:
    from ..config import cache_root

    return cache_root() / "series" / BOOSTED_FOLDER / f"{name}.joblib"


class Boosted:
    name = "gradient boosting"
    dated = False
    fitted = ("model", "spread")

    def __init__(self, effort: int = 0) -> None:
        self.effort = effort

    def fit(self, groups: list[Group]) -> None:
        import joblib

        from .checkpoint import key

        x, y, late = table(groups, self.dated)
        known = ~np.isnan(y)
        x, y, late = x[known], y[known], late[known]
        iterations = ITERATIONS * 2**self.effort
        cached = _boosted_path(key(self.name, SEED, arrays=[x, y, late]))
        kept = joblib.load(cached) if cached.exists() else None
        if kept and (iterations == kept["iterations"] or kept["converged"] and iterations > kept["iterations"]):
            print(f"  {self.name}: same fit as before ({kept['used']} of {kept['iterations']} iterations), reusing it")
            for name in self.fitted:
                setattr(self, name, kept[name])
            return
        print(f"  {self.name}: fitting {len(y):,} rows of {x.shape[1]} features, at most {iterations} iterations")
        self._fit(x, y, late, iterations, max(g.horizon for g in groups))
        used = max(self.used)
        cached.parent.mkdir(parents=True, exist_ok=True)
        state = {name: getattr(self, name) for name in self.fitted} | {"used": used, "iterations": iterations}
        joblib.dump({**state, "converged": used < iterations}, cached)

    def _fit(self, x: np.ndarray, y: np.ndarray, late: np.ndarray, iterations: int, longest: int) -> None:
        from sklearn.ensemble import HistGradientBoostingRegressor

        model = HistGradientBoostingRegressor(loss="squared_error", max_iter=iterations, random_state=SEED)
        self.model = model.fit(x[~late], y[~late])
        self.used = [self.model.n_iter_]
        steps = x[late][:, STEP_COLUMN].astype(int)
        errors = y[late] - self.model.predict(x[late])
        spread = [
            np.quantile(errors[steps == k], QUANTILES) if (steps == k).sum() > 2 else np.zeros(len(QUANTILES))
            for k in range(longest)
        ]
        self.spread = np.asarray(spread).T
        self.model.fit(x, y)
        self.used.append(self.model.n_iter_)

    def _levels(self, x: np.ndarray, n: int, horizon: int) -> np.ndarray:
        return self.model.predict(x).reshape(n, horizon)[:, None] + self.spread[None, :, :horizon]

    def predict_many(self, histories: list[np.ndarray], season: int, horizon: int, clocks: Clocks = None) -> list:
        x, base, scales, noise = forecast_rows(histories, season, horizon, clocks, self.dated)
        levels = self._levels(x, len(histories), horizon)
        return list(np.sort((base[:, None] + levels * noise[:, None, None]) * scales[:, None, None], axis=1))


class QuantileBoosted(Boosted):
    name = "gradient boosting, quantile"
    dated = True
    fitted = ("models",)

    def _fit(self, x: np.ndarray, y: np.ndarray, late: np.ndarray, iterations: int, longest: int) -> None:
        from sklearn.ensemble import HistGradientBoostingRegressor

        def one(q: float) -> Any:
            return HistGradientBoostingRegressor(loss="quantile", quantile=q, max_iter=iterations, random_state=SEED)

        self.models = [one(q).fit(x, y) for q in QUANTILES]
        self.used = [m.n_iter_ for m in self.models]

    def _levels(self, x: np.ndarray, n: int, horizon: int) -> np.ndarray:
        return np.stack([m.predict(x).reshape(n, horizon) for m in self.models], axis=1)


def rivals() -> list[Any]:
    out: list[Any] = [SeasonalNaive()]
    try:
        from .timestral_rival import Timestral

        out.append(Timestral())
    except ImportError:
        pass
    return out
