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


def pinball(future: np.ndarray, quantiles: np.ndarray) -> np.ndarray:
    keep = ~np.isnan(future)
    y, q = future[keep], quantiles[:, keep]
    return 2 * np.sum(np.abs((y - q) * ((y <= q) - np.asarray(QUANTILES)[:, None])), axis=1)


def crps(future: np.ndarray, quantiles: np.ndarray) -> tuple[float, float]:
    total = float(np.sum(np.abs(future[~np.isnan(future)])))
    return float(pinball(future, quantiles).sum() / len(QUANTILES)), total


def summary(windows: list[dict[str, Any]], forecasts: list[np.ndarray], seasons: list[int]) -> dict[str, Any]:
    mases, losses, totals = [], 0.0, 0.0
    for w, q, s in zip(windows, forecasts, seasons, strict=True):
        mases.append(mase(w["history"], w["future"], q[QUANTILES.index(0.5)], s))
        loss, total = crps(w["future"], q)
        losses, totals = losses + loss, totals + total
    return {"mase": float(np.nanmean(mases)), "crps": losses / totals if totals else float("nan"), "windows": mases}


def relative(scores: dict[str, float], naive: dict[str, float]) -> dict[str, float]:
    return {k: scores[k] / naive[k] for k in scores if naive.get(k)}


def geomean(values: list[float]) -> float:
    arr = np.asarray([v for v in values if v and np.isfinite(v)])
    return float(np.exp(np.mean(np.log(arr)))) if len(arr) else float("nan")


def ratio_interval(ours: list[list[float]], base: list[list[float]], draws: int, seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    pairs = [(np.asarray(o), np.asarray(b)) for o, b in zip(ours, base, strict=True)]
    samples = []
    for _ in range(draws):
        ratios = []
        for o, b in pairs:
            pick = rng.integers(0, len(o), len(o))
            ratios.append(np.nanmean(o[pick]) / np.nanmean(b[pick]))
        samples.append(geomean(ratios))
    return [float(np.nanpercentile(samples, 2.5)), float(np.nanpercentile(samples, 97.5))]
