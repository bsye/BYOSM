from __future__ import annotations

import math
import random
from typing import Any

from ..evals.selection import selection_value
from ..project import Project
from ..spec import TaskSpec
from .techniques import eligible, value_at

GAMMA = 0.25
CANDIDATES = 24
MIN_TRIALS = 4
MIN_BANDWIDTH = 0.05
TPE = "tpe"
TPE_PREFIX = "tpe-"
SEARCHED = "train."


def numeric(v: Any) -> bool:
    return isinstance(v, int | float) and not isinstance(v, bool)


def numeric_paths(spec: TaskSpec) -> list[str]:
    out = []
    for pattern in (pattern for t in eligible(spec) for pattern in t.paths):
        head, _, tail = pattern.rpartition(".")
        if not pattern.startswith(SEARCHED) or "*" in head or "=" in pattern:
            continue
        out += [path for k in tail.split("|") if numeric(value_at(spec, path := f"{head}.{k}"))]
    return out


def trials(p: Project, path: str) -> list[tuple[float, float]]:
    board, prim = p.board(), p.spec().primary_suite()
    out = []
    for e in p.experiments():
        score = selection_value(board.get(e, {}).get(prim))
        edited = any(o["path"] == path for a in (p.proposal(e) or {}).get("applied", []) for o in a.get("ops", []))
        if score is not None and (edited or e == p.promoted):
            out.append((float(value_at(p.exp_spec(e), path)), score))
    return out


def _density(x: float, points: list[float], bw: float) -> float:
    return sum(math.exp(-((x - c) ** 2) / (2 * bw * bw)) for c in points) / len(points) + 1e-12


def suggest(tried: list[tuple[float, float]], integer: bool, rng: random.Random) -> float | None:
    if len({v for v, _ in tried}) < MIN_TRIALS:
        return None
    log = all(v > 0 for v, _ in tried)
    xs = [(math.log(v) if log else v, s) for v, s in sorted(tried, key=lambda t: -t[1])]
    cut = max(1, math.ceil(GAMMA * len(xs)))
    good, bad = [x for x, _ in xs[:cut]], [x for x, _ in xs[cut:]]
    span = max(x for x, _ in xs) - min(x for x, _ in xs)
    bw = max(span / len(xs), MIN_BANDWIDTH)
    cands = [rng.gauss(rng.choice(good), bw) for _ in range(CANDIDATES)]
    best = max(cands, key=lambda c: _density(c, good, bw) / _density(c, bad, bw))
    value = math.exp(best) if log else best
    return max(1, round(value)) if integer else float(f"{value:.3g}")


def tpe_proposals(p: Project, spec: TaskSpec) -> list[dict]:
    rng = random.Random(f"{p.name}:{len(p.attempts())}")
    out = []
    for path in numeric_paths(spec):
        tried = trials(p, path)
        value = suggest(tried, isinstance(value_at(spec, path), int), rng)
        if value is None:
            continue
        best = max(tried, key=lambda t: t[1])
        why = (
            f"TPE over {len(tried)} tried values of {path} (best {best[0]:g}, calib {best[1]:.4f}) suggests {value:g}."
        )
        op = {"op": "set", "path": path, "value": value}
        out.append({"id": f"{TPE_PREFIX}{path}", "why": why, "ops": [op], "source": TPE})
    return out
