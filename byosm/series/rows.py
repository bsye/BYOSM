from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .calendar import Clock, features

if TYPE_CHECKING:
    from .models import Group

MAX_LAGS = 96
ORIGINS_PER_SERIES = 40
MAX_ROWS = 1_500_000
CALIB_SHARE = 0.1
SCALE_SEASONS = 7
MIN_NOISE = 1e-3
SEASONAL_COLUMN = -3
STEP_COLUMN = -2
MAX_UNITS = 10.0
Table = tuple[np.ndarray, np.ndarray, np.ndarray]


def level(history: np.ndarray, season: int) -> float:
    tail = history[-max(SCALE_SEASONS * season, 8) :]
    tail = tail[~np.isnan(tail)]
    value = float(np.mean(np.abs(tail))) if len(tail) else 1.0
    return value or 1.0


def noise(history: np.ndarray, season: int, scale: float) -> float:
    tail = history[-max(SCALE_SEASONS * season, 8) - season :]
    diffs = np.abs(tail[season:] - tail[:-season]) if len(tail) > season else np.zeros(0)
    diffs = diffs[~np.isnan(diffs)]
    return max(float(diffs.mean()) / scale if len(diffs) else 1.0, MIN_NOISE)


def _lags(season: int) -> int:
    return min(MAX_LAGS, max(2 * season, 24))


def rows(
    history: np.ndarray, origin: int, horizon: int, scale: float, season: int, calendar: np.ndarray | None
) -> np.ndarray:
    lags = history[max(0, origin - MAX_LAGS) : origin] / scale
    lags = np.concatenate([np.full(MAX_LAGS - len(lags), np.nan), lags])
    recent = history[max(0, origin - season) : origin] / scale
    present = recent[~np.isnan(recent)]
    stats = [present.mean(), present.std(), present.min(), present.max()] if len(present) else [np.nan] * 4
    steps = np.arange(horizon)
    back = origin - season + steps % season
    seasonal = np.where((back >= 0) & (back < origin), history[np.clip(back, 0, len(history) - 1)] / scale, np.nan)
    per_step = np.column_stack([seasonal, steps, (origin + steps) % season])
    shared = np.tile(np.concatenate([lags, stats, [season]]), (horizon, 1))
    return np.hstack([shared, per_step] if calendar is None else [shared, calendar, per_step])


def base(x: np.ndarray) -> np.ndarray:
    seasonal, last = x[:, SEASONAL_COLUMN], x[:, MAX_LAGS - 1]
    return np.nan_to_num(np.where(np.isnan(seasonal), last, seasonal))


def _calendar(at: Clock | None, n: int, dated: bool) -> np.ndarray | None:
    return features(at, n) if dated else None


def _series_rows(h: np.ndarray, at: Clock | None, g: Group, per_series: int, dated: bool) -> list[tuple]:
    start, usable = _lags(g.season) // 2, len(h) - g.horizon
    if usable <= start:
        return []
    origins = np.unique(np.linspace(start, usable - 1, min(per_series, usable)).astype(int))
    cut = origins[int(len(origins) * (1 - CALIB_SHARE))] if len(origins) > 1 else usable
    calendar, out = _calendar(at, len(h) + g.horizon, dated), []
    for origin in origins:
        scale = level(h[:origin], g.season)
        x = rows(h, origin, g.horizon, scale, g.season, None if calendar is None else calendar[origin:][: g.horizon])
        target = (h[origin : origin + g.horizon] / scale - base(x)) / noise(h[:origin], g.season, scale)
        out.append((x, np.clip(target, -MAX_UNITS, MAX_UNITS), np.full(g.horizon, origin >= cut)))
    return out


def table(groups: list[Group], dated: bool) -> Table:
    found = []
    for g in groups:
        per_series = max(ORIGINS_PER_SERIES, MAX_ROWS // len(groups) // max(1, len(g.histories) * g.horizon))
        clocks = g.clocks or [None] * len(g.histories)
        for h, at in zip(g.histories, clocks, strict=True):
            found += _series_rows(h, at, g, per_series, dated)
    xs, ys, late = zip(*found, strict=True)
    return np.vstack(xs), np.concatenate(ys), np.concatenate(late)


def forecast_rows(
    histories: list[np.ndarray], season: int, horizon: int, clocks: list[Clock | None] | None, dated: bool
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    scales = np.asarray([level(h, season) for h in histories])
    at = clocks or [None] * len(histories)
    calendars = [_calendar(c, len(h) + horizon, dated) for h, c in zip(histories, at, strict=True)]
    x = np.vstack(
        [
            rows(h, len(h), horizon, sc, season, None if cal is None else cal[len(h) :])
            for h, sc, cal in zip(histories, scales, calendars, strict=True)
        ]
    )
    spread = np.asarray([noise(h, season, sc) for h, sc in zip(histories, scales, strict=True)])
    return x, base(x).reshape(len(histories), horizon), scales, spread
