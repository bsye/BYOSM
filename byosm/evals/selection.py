from __future__ import annotations

from .. import metrics
from ..project import Project
from ..spec import TaskSpec

SURE_LEAD = 0.95


def selection_value(r: dict | None) -> float | None:
    if not r:
        return None
    return r.get("calib_objective", r.get("objective"))


def compare(spec: TaskSpec, a: dict | None, b: dict | None, split: str = "calib") -> dict[str, float] | None:
    ua, ub = (a or {}).get("units", {}).get(split), (b or {}).get("units", {}).get(split)
    if not ua or not ub:
        return None
    try:
        return metrics.bootstrap(spec, ua, ub)
    except ValueError:
        return None


def calibrated_threshold(p: Project, exp: str) -> float | None:
    r = p.board().get(exp, {}).get(p.spec().primary_suite()) or {}
    t = r.get("metrics", {}).get("threshold")
    return t if t is not None else p.meta(exp).get("threshold")


def sure_lead(cmp: dict | None) -> bool:
    return cmp is None or cmp["p_better"] >= SURE_LEAD
