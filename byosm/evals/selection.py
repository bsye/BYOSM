from __future__ import annotations

from .. import metrics
from ..project import Project
from ..spec import TaskSpec

SURE_LEAD = 0.95


def selection_value(r: dict | None) -> float | None:
    if not r:
        return None
    return r.get("calib_objective", r.get("objective"))


def compare_runs(spec: TaskSpec, a: list[dict], b: list[dict], split: str = "calib") -> dict[str, float] | None:
    ua = [u for r in a if (u := (r or {}).get("units", {}).get(split))]
    ub = [u for r in b if (u := (r or {}).get("units", {}).get(split))]
    if not ua or not ub:
        return None
    try:
        return metrics.bootstrap_seeds(spec, ua, ub)
    except ValueError:
        return None


def compare(spec: TaskSpec, a: dict | None, b: dict | None, split: str = "calib") -> dict[str, float] | None:
    return compare_runs(spec, [a] if a else [], [b] if b else [], split)


def calibrated_threshold(p: Project, exp: str) -> float | None:
    r = p.board().get(exp, {}).get(p.spec().primary_suite()) or {}
    t = r.get("metrics", {}).get("threshold")
    return t if t is not None else p.meta(exp).get("threshold")


def clean_selection(p: Project, spec: TaskSpec, exp: str) -> bool:
    prim = spec.primary_suite()
    kind = next((e.kind for e in spec.evals if e.name == prim), "builtin")
    return kind != "builtin" or bool(p.meta(exp).get("calib_held"))


def sure_lead(cmp: dict | None) -> bool:
    return cmp is None or cmp["p_better"] >= SURE_LEAD
