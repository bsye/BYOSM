from __future__ import annotations

from typing import Any

from ..health_score import CHECK_TITLES, DROP_OK, DROP_WARN, UNKNOWN, check
from ..project import Project
from ..spec import TaskSpec
from . import overfit_checks as checks

FRESH = "fresh"
WRITTEN = "generated"
CHECK_WEIGHT = 2
TITLE = CHECK_TITLES["overfit"]


def required(spec: TaskSpec) -> bool:
    return spec.task_type == "classify"


def written_suites(base: TaskSpec, results: dict[str, dict]) -> list[str]:
    prim = base.primary_suite()
    return [e.name for e in base.evals if e.kind == WRITTEN and e.name != prim and e.name in results]


def finding(base: TaskSpec, results: dict[str, dict]) -> list[dict[str, Any]]:
    written = written_suites(base, results)
    if not written:
        return []
    suite = max(written, key=lambda s: results[s].get("n_test") or 0)
    test, new = results[base.primary_suite()]["objective"], results[suite]["objective"]
    drop = test - new
    sev = "fail" if drop > DROP_WARN else "warn" if drop > DROP_OK else "ok"
    text = {
        "ok": "On test cases written after training the model scores about as on its own test set.",
        "warn": "On test cases written after training the model scores clearly lower: part of its score comes from quirks of its own benchmark.",
        "fail": "On test cases written after training the model scores far lower: its benchmark score does not carry over to new inputs.",
    }[sev]
    nums = {
        "test": test,
        "new_cases": new,
        "drop": drop,
        "rows": results[suite].get("n_test"),
        "warn_above": DROP_OK,
    }
    return [checks.finding(FRESH, sev, text, nums, suite=suite)]


def overfit_health(p: Project, spec: TaskSpec, report: dict | None) -> dict:
    if not report or report.get("experiment") != p.promoted:
        why = "No overfitting check for the current model."
        return check("overfit", TITLE, CHECK_WEIGHT, UNKNOWN, "not checked", why, ["overfit", p.name])
    if required(spec) and not any(f.get("check") == FRESH for f in report.get("findings", [])):
        scored = written_suites(spec, p.board().get(p.promoted or "", {}))
        why = "Not yet scored on test cases written after training, so memorised benchmark quirks would go unnoticed. "
        why += "Runs the check again with them." if scored else "Writes a new test set the model never saw."
        args = ["overfit", p.name] if scored else ["testset", p.name]
        return check("overfit", TITLE, CHECK_WEIGHT, UNKNOWN, "no new test cases", why, args)
    return check("overfit", TITLE, CHECK_WEIGHT, report["verdict"], report["verdict"], report["summary"])
