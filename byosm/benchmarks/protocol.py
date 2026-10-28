from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from ..project import Project
from ..spec import EvalSuite, TaskSpec

TOLERANCE = 5e-5
REPORT_PREFIX = "protocol:"
Values = dict[str, float | None]
Published = dict[str, Values]


@dataclass(frozen=True)
class Kind:
    fits: Callable[[TaskSpec], bool]
    expected: Callable[[TaskSpec, list[dict]], Values]
    observed: Callable[[dict[str, Any]], Values]
    published: Callable[[TaskSpec], Published] | None = None


def same(a: float | None, b: float | None) -> bool:
    return a is b if a is None or b is None else abs(a - b) <= TOLERANCE


def compared(expected: Values, observed: Values) -> list[dict[str, Any]]:
    return [
        {"metric": k, "expected": v, "scored": observed.get(k), "ok": same(v, observed.get(k))}
        for k, v in expected.items()
    ]


def run(
    p: Project, spec: TaskSpec, suite: EvalSuite, kind: Kind, rows: list[dict], log: Callable[[str], None]
) -> dict[str, Any]:
    from ..evals.scoring import score
    from ..evals.suites import materialize
    from ..predictors import ALWAYS_POSITIVE, get_baseline

    expected = kind.expected(spec, rows)
    calib, test, root = materialize(p, spec, suite, log)
    found: dict[str, Any] = {"baseline": ALWAYS_POSITIVE, "n_test": len(test), "n_calib": len(calib), "checks": []}
    if test:
        r = score(get_baseline(ALWAYS_POSITIVE, spec), spec, calib, test, root, suite.label_space(spec.label_names))
        found["checks"] = compared(expected, kind.observed(r))
    found["passed"] = bool(found["checks"]) and all(c["ok"] for c in found["checks"])
    if found["passed"] and kind.published:
        found["published"] = kind.published(spec)
    p.save_report(REPORT_PREFIX + suite.name, found)
    return found
