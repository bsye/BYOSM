from __future__ import annotations

from collections.abc import Callable

import numpy as np

Objective = Callable[[np.ndarray], float]


def _draws(sa: np.ndarray, sb: np.ndarray | None, objective: Objective, n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        idx = rng.integers(0, len(sa), len(sa))
        va = objective(sa[idx].sum(0))
        vals.append(va - objective(sb[idx].sum(0)) if sb is not None else va)
    return np.asarray(vals)


def bootstrap_stats(
    sa: np.ndarray, sb: np.ndarray | None, objective: Objective, n: int = 1000, seed: int = 0
) -> dict[str, float]:
    if sb is not None and sb.shape != sa.shape:
        raise ValueError("paired bootstrap needs the same units")
    v = _draws(sa, sb, objective, n, seed)
    lo, hi = np.quantile(v, [0.025, 0.975])
    out = {"lo": float(lo), "hi": float(hi)}
    if sb is not None:
        out.update(
            delta=objective(sa.sum(0)) - objective(sb.sum(0)),
            p_better=float((v > 0).mean() + 0.5 * (v == 0).mean()),
        )
    return out
