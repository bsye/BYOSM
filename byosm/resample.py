from __future__ import annotations

from collections.abc import Callable

import numpy as np

Objective = Callable[[np.ndarray], float]


Runs = list[np.ndarray]


def _mean(runs: Runs, objective: Objective, idx: np.ndarray | None = None) -> float:
    return float(np.mean([objective((r if idx is None else r[idx]).sum(0)) for r in runs]))


def _draws(sa: Runs, sb: Runs | None, objective: Objective, n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        idx = rng.integers(0, len(sa[0]), len(sa[0]))
        va = _mean(sa, objective, idx)
        vals.append(va - _mean(sb, objective, idx) if sb is not None else va)
    return np.asarray(vals)


def bootstrap_runs(sa: Runs, sb: Runs | None, objective: Objective, n: int = 1000, seed: int = 0) -> dict[str, float]:
    if sb is not None and len({r.shape for r in sa + sb}) > 1:
        raise ValueError("paired bootstrap needs the same units")
    v = _draws(sa, sb, objective, n, seed)
    lo, hi = np.quantile(v, [0.025, 0.975])
    out = {"lo": float(lo), "hi": float(hi)}
    if sb is not None:
        out.update(
            delta=_mean(sa, objective) - _mean(sb, objective),
            p_better=float((v > 0).mean() + 0.5 * (v == 0).mean()),
        )
    return out


def bootstrap_suites(pairs: list[tuple[Runs, Runs]], objective: Objective, n: int = 1000, seed: int = 0) -> dict:
    if any(len({r.shape for r in sa + sb}) > 1 for sa, sb in pairs):
        raise ValueError("paired bootstrap needs the same units")
    v = np.mean([_draws(sa, sb, objective, n, seed + i) for i, (sa, sb) in enumerate(pairs)], axis=0)
    lo, hi = np.quantile(v, [0.025, 0.975])
    delta = float(np.mean([_mean(sa, objective) - _mean(sb, objective) for sa, sb in pairs]))
    return {"lo": float(lo), "hi": float(hi), "delta": delta, "p_better": float((v > 0).mean() + 0.5 * (v == 0).mean())}


def bootstrap_stats(
    sa: np.ndarray, sb: np.ndarray | None, objective: Objective, n: int = 1000, seed: int = 0
) -> dict[str, float]:
    return bootstrap_runs([sa], None if sb is None else [sb], objective, n, seed)
