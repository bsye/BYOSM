from __future__ import annotations

import csv
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from .pairs import FITS

THRESHOLD = 0.5
BOOTSTRAP = 1000
SEED = 0


def groups(table: Path, split: str = "test") -> dict[str, list[dict[str, str]]]:
    out: dict[str, list[dict[str, str]]] = {}
    with table.open(newline="") as f:
        for row in csv.DictReader(f):
            if row["split"] == split:
                out.setdefault(row["group"], []).append(row)
    return out


def decide(chances: list[float], threshold: float = THRESHOLD) -> set[int]:
    return {i for i, p in enumerate(chances) if p >= threshold}


def _ci(hits: np.ndarray) -> tuple[float, float]:
    if not len(hits):
        return 0.0, 0.0
    rng = np.random.default_rng(SEED)
    means = [hits[rng.integers(0, len(hits), len(hits))].mean() for _ in range(BOOTSTRAP)]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def _share(hits: list[bool]) -> dict[str, Any]:
    arr = np.asarray(hits, dtype=float)
    low, high = _ci(arr)
    return {
        "accuracy": round(float(arr.mean()), 4) if len(arr) else None,
        "ci": [round(low, 4), round(high, 4)],
        "n": len(arr),
    }


def score(table: Path, chance: Callable[[list[str]], list[float]], threshold: float = THRESHOLD) -> dict[str, Any]:
    exact, none, some = [], [], []
    for rows in groups(table).values():
        truth = {i for i, r in enumerate(rows) if r["label"] == FITS}
        hit = decide(chance([r["text"] for r in rows]), threshold) == truth
        exact.append(hit)
        (some if truth else none).append(hit)
    return {"threshold": threshold, "exact": _share(exact), "tool_needed": _share(some), "no_tool": _share(none)}
