from __future__ import annotations

from typing import Any

import numpy as np

QUANTILES = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)


def _scale(history: np.ndarray, season: int) -> float:
    h = history[~np.isnan(history)]
    step = season if len(h) > season else 1
    diffs = np.abs(h[step:] - h[:-step])
    return float(np.mean(diffs)) if len(diffs) and np.mean(diffs) > 0 else 1.0


def mase(history: np.ndarray, future: np.ndarray, median: np.ndarray, season: int) -> float:
    keep = ~np.isnan(future)
    return float(np.mean(np.abs(future[keep] - median[keep])) / _scale(history, season)) if keep.any() else np.nan


def crps(future: np.ndarray, quantiles: np.ndarray) -> tuple[float, float]:
    keep = ~np.isnan(future)
    y, q = future[keep], quantiles[:, keep]
    loss = sum(2 * np.sum(np.abs((y - q[i]) * ((y <= q[i]) - level))) for i, level in enumerate(QUANTILES))
    return float(loss / len(QUANTILES)), float(np.sum(np.abs(y)))


def summary(windows: list[dict[str, Any]], forecasts: list[np.ndarray], seasons: list[int]) -> dict[str, float]:
    mases, losses, totals = [], 0.0, 0.0
    for w, q, s in zip(windows, forecasts, seasons, strict=True):
        mases.append(mase(w["history"], w["future"], q[QUANTILES.index(0.5)], s))
        loss, total = crps(w["future"], q)
        losses, totals = losses + loss, totals + total
    return {"mase": float(np.nanmean(mases)), "crps": losses / totals if totals else float("nan")}


def relative(scores: dict[str, float], naive: dict[str, float]) -> dict[str, float]:
    return {k: scores[k] / naive[k] for k in scores if naive.get(k)}


def geomean(values: list[float]) -> float:
    arr = np.asarray([v for v in values if v and np.isfinite(v)])
    return float(np.exp(np.mean(np.log(arr)))) if len(arr) else float("nan")
