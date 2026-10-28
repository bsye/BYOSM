from __future__ import annotations

import math
from collections import defaultdict
from statistics import mean, stdev
from typing import Any

from scipy.stats import t

from .. import evals
from ..project import Project
from ..spec import TaskSpec

EFFECT = "effect:"
LEVEL = 0.95
MIN_MEASURED = 2
DIGITS = 4
IDENTITY_KEYS = ("fingerprint", "identity")
EXPERIMENT_FIELDS = ("specialist", "experiment", "parent", "delta", "lo", "hi")


def _rounded(x: float | None) -> float | None:
    return None if x is None else round(float(x), DIGITS)


def _comparable(mine: dict | None, theirs: dict | None) -> bool:
    if not mine or not theirs or mine.get("calib_objective") is None or theirs.get("calib_objective") is None:
        return False
    return mine.get("fingerprint") is not None and mine.get("fingerprint") == theirs.get("fingerprint")


def _key(parent: str, mine: dict, theirs: dict) -> str:
    return "|".join([parent, *(str(r.get(k)) for r in (mine, theirs) for k in IDENTITY_KEYS)])


def _measure(spec: TaskSpec, parent: str, mine: dict, theirs: dict) -> dict[str, Any]:
    cmp = evals.compare(spec, mine, theirs) or {}
    return {
        "key": _key(parent, mine, theirs),
        "parent": parent,
        "delta": _rounded(mine["calib_objective"] - theirs["calib_objective"]),
        "lo": _rounded(cmp.get("lo")),
        "hi": _rounded(cmp.get("hi")),
    }


def _delta_only(parent: str, mine: dict, theirs: dict) -> dict[str, Any]:
    delta = _rounded(mine["calib_objective"] - theirs["calib_objective"])
    return {"key": _key(parent, mine, theirs), "parent": parent, "delta": delta, "lo": None, "hi": None}


def effects(p: Project, parents: dict[str, str | None], bootstrap: bool = True) -> dict[str, dict[str, Any]]:
    spec, board, cache = p.spec(), p.board(), p.reports(EFFECT)
    suite = spec.primary_suite()
    out: dict[str, dict[str, Any]] = {}
    for exp, parent in parents.items():
        mine, theirs = board.get(exp, {}).get(suite), board.get(parent or "", {}).get(suite)
        if not parent or not _comparable(mine, theirs):
            continue
        found = cache.get(EFFECT + exp) or {}
        if found.get("key") != _key(parent, mine, theirs):
            found = _measure(spec, parent, mine, theirs) if bootstrap else _delta_only(parent, mine, theirs)
            if bootstrap:
                p.save_report(EFFECT + exp, found)
        out[exp] = found
    return out


def measured(p: Project, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    spec = p.spec()
    found = effects(p, {r["experiment"]: r["parent"] for r in rows}, bootstrap=False)
    kind = {"specialist": p.name, "modality": spec.modality, "task_type": spec.task_type}
    return [{**kind, **r, **found[r["experiment"]]} for r in rows if r["experiment"] in found]


def summary(deltas: list[float]) -> dict[str, Any]:
    n = len(deltas)
    out = {"measured": n, "mean": _rounded(mean(deltas)) if n else None, "lo": None, "hi": None}
    if n >= MIN_MEASURED:
        half = t.ppf((1 + LEVEL) / 2, n - 1) * stdev(deltas) / math.sqrt(n)
        out.update(lo=_rounded(mean(deltas) - half), hi=_rounded(mean(deltas) + half))
    return out


def family_effect(found: list[dict[str, Any]]) -> dict[str, Any]:
    kinds: dict[tuple[str, str], list[float]] = defaultdict(list)
    for f in found:
        kinds[(f["modality"], f["task_type"])].append(f["delta"])
    return {
        **summary([f["delta"] for f in found]),
        "kinds": [{"modality": m, "task_type": k, **summary(ds)} for (m, k), ds in sorted(kinds.items())],
        "experiments": [{k: f[k] for k in EXPERIMENT_FIELDS} for f in found],
    }
