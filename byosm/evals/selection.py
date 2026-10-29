from __future__ import annotations

from .. import metrics
from ..project import Project
from ..spec import TaskSpec

SURE_LEAD = 0.95


def selection_value(r: dict | None) -> float | None:
    if not r:
        return None
    return r.get("calib_objective", r.get("objective"))


def selection_mean(by_suite: dict, suites: list[str]) -> float | None:
    values = [selection_value(by_suite.get(s)) for s in suites]
    return None if not values or None in values else sum(values) / len(values)


def selected(spec: TaskSpec, by_suite: dict) -> list[str]:
    from ..data.sufficiency import MIN_CALIB_ITEMS

    prim = spec.primary_suite()
    return [s for s in spec.selecting() if s == prim or (by_suite.get(s, {}).get("n_calib") or 0) >= MIN_CALIB_ITEMS]


def _units(rs: list[dict], split: str) -> list[dict]:
    return [u for r in rs if (u := (r or {}).get("units", {}).get(split))]


def compare_suites(
    spec: TaskSpec, a: dict[str, list[dict]], b: dict[str, list[dict]], split: str = "calib"
) -> dict | None:
    pairs = [(_units(a[s], split), _units(b.get(s, []), split)) for s in a]
    if not pairs or not all(ua and ub for ua, ub in pairs):
        return None
    try:
        return metrics.bootstrap_suite_seeds(spec, pairs)
    except ValueError:
        return None


def compare_runs(spec: TaskSpec, a: list[dict], b: list[dict], split: str = "calib") -> dict[str, float] | None:
    return compare_suites(spec, {"": a}, {"": b}, split)


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
