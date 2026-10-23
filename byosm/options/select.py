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
REPORT_PREFIX = "options:"


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


def sampled(table: Path, sample: int | None = None) -> dict[str, list[dict[str, str]]]:
    found = groups(table)
    keys = sorted(found)
    if sample and sample < len(keys):
        keys = sorted(np.random.default_rng(SEED).choice(keys, sample, replace=False).tolist())
    return {k: found[k] for k in keys}


def evaluate(
    found: dict[str, list[dict[str, str]]], choose: Callable[[str, list[dict[str, str]]], set[int]]
) -> dict[str, Any]:
    exact, none, some = [], [], []
    for key, rows in found.items():
        truth = {i for i, r in enumerate(rows) if r["label"] == FITS}
        hit = choose(key, rows) == truth
        exact.append(hit)
        (some if truth else none).append(hit)
    return {"exact": _share(exact), "tool_needed": _share(some), "no_tool": _share(none)}


def score(
    table: Path, chance: Callable[[list[str]], list[float]], threshold: float = THRESHOLD, sample: int | None = None
) -> dict[str, Any]:
    result = evaluate(sampled(table, sample), lambda _, rows: decide(chance([r["text"] for r in rows]), threshold))
    return {"threshold": threshold, **result}


def score_export(ws: Path, name: str, threshold: float = THRESHOLD) -> dict[str, Any]:
    import time

    from ..db import get_or_add, session
    from ..db.models import WorkspaceReport
    from ..project import Project
    from ..rt import load
    from .create import table_path

    model = load(Project(name).root / "export", threads=4)
    result = {
        "at": time.time(),
        **score(table_path(ws, name), lambda texts: [model.predict(t)["probs"][FITS] for t in texts], threshold),
    }
    with session(ws) as s:
        get_or_add(s, WorkspaceReport, kind=REPORT_PREFIX + name).data = result
    return result


def report(ws: Path, name: str) -> dict[str, Any] | None:
    from sqlalchemy import select

    from ..db import session
    from ..db.models import WorkspaceReport

    with session(ws) as s:
        return s.scalar(select(WorkspaceReport.data).where(WorkspaceReport.kind == REPORT_PREFIX + name))
