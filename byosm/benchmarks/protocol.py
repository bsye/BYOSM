from __future__ import annotations

from collections.abc import Callable
from typing import Any

from ..evals.grouped import GROUP, PAIR_LABELS, REQUESTS
from ..options.situations import FITS
from ..project import Project
from ..spec import EvalSuite, TaskSpec
from .pairs import TEST

TOLERANCE = 5e-5
REPORT_PREFIX = "protocol:"
Values = dict[str, float | None]


def pairs_fit(spec: TaskSpec) -> bool:
    return set(spec.label_names) == PAIR_LABELS and spec.label_names[spec.pos_index] == FITS


def pair_expected(spec: TaskSpec, rows: list[dict]) -> Values:
    test = [r for r in rows if r["split"] == TEST]
    share = sum(r["label"] == FITS for r in test) / max(1, len(test))
    requests: dict[str, list[bool]] = {}
    for r in test:
        requests.setdefault(r["meta"][GROUP], []).append(r["label"] == FITS)
    needed = [all(found) for found in requests.values() if any(found)]
    return {
        "precision": share,
        "recall": 1.0 if share else 0.0,
        "f1": 2 * share / (1 + share),
        "tool_needed": sum(needed) / len(needed) if needed else None,
        "no_tool": 0.0 if len(needed) < len(requests) else None,
    }


def pair_observed(result: dict[str, Any]) -> Values:
    m = result["metrics"]
    found = {k: m[REQUESTS]["all"][k]["accuracy"] for k in ("tool_needed", "no_tool")}
    return {k: m[k] for k in ("precision", "recall", "f1")} | found


def same(a: float | None, b: float | None) -> bool:
    return a is b if a is None or b is None else abs(a - b) <= TOLERANCE


def compared(expected: Values, observed: Values) -> list[dict[str, Any]]:
    return [
        {"metric": k, "expected": v, "scored": observed.get(k), "ok": same(v, observed.get(k))}
        for k, v in expected.items()
    ]


def run(
    p: Project,
    spec: TaskSpec,
    suite: EvalSuite,
    expected: Values,
    observe: Callable[[dict], Values],
    log: Callable[[str], None],
) -> dict[str, Any]:
    from ..evals.scoring import score
    from ..evals.suites import materialize
    from ..predictors import ALWAYS_POSITIVE, get_baseline

    calib, test, root = materialize(p, spec, suite, log)
    if not test:
        return {"baseline": ALWAYS_POSITIVE, "n_test": 0, "n_calib": len(calib), "checks": [], "passed": False}
    r = score(get_baseline(ALWAYS_POSITIVE, spec), spec, calib, test, root, suite.label_space(spec.label_names))
    checks = compared(expected, observe(r))
    found = {"baseline": ALWAYS_POSITIVE, "n_test": r["n_test"], "n_calib": r["n_calib"], "checks": checks}
    found["passed"] = all(c["ok"] for c in checks)
    p.save_report(REPORT_PREFIX + suite.name, found)
    return found
