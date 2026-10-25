from __future__ import annotations

from typing import Any

from .evals.overfit import OVERFIT
from .health_score import OK, UNKNOWN
from .project import Project
from .spec import TaskSpec

GAP, CURVE = "gap", "curve"
FIT_PREFIX = "fit-"
LR_FACTOR = 0.5


def _check(cs: list[dict], cid: str) -> dict | None:
    return next((c for c in cs if c["id"] == cid and c["status"] not in (OK, UNKNOWN)), None)


def _finding(report: dict, check: str) -> dict:
    return next((f for f in report.get("findings", []) if f["check"] == check and f["severity"] != OK), {})


def less_overfit(p: Project, spec: TaskSpec, cs: list[dict]) -> dict[str, Any] | None:
    report = p.report(OVERFIT) or {}
    exp = f"{FIT_PREFIX}{p.promoted}"
    if not _check(cs, "overfit") or p.experiment(exp) is not None:
        return None
    gap, curve = _finding(report, GAP).get("numbers", {}), _finding(report, CURVE).get("numbers", {})
    lr = spec.train.lr_backbone * LR_FACTOR
    sets = [f"train.lr_backbone={lr:g}"]
    why = [f"Halve the encoder learning rate ({spec.train.lr_backbone:g} to {lr:g})"]
    if curve.get("best_epoch") and curve["best_epoch"] < spec.train.epochs:
        sets.append(f"train.epochs={curve['best_epoch']}")
        why.append(f"stop at epoch {curve['best_epoch']}, where dev peaked")
    seen = (
        f"Train scores {gap['train']:.3f} but test {gap['test']:.3f}: it memorised its training rows. "
        if gap.get("train") is not None
        else "The overfitting check failed. "
    )
    args = ["exp", "create", p.name, exp, "--base", str(p.promoted), *(a for s in sets for a in ("--set", s)), "--run"]
    return {"id": "fix-overfit", "title": "Reduce overfitting", "why": seen + " and ".join(why) + ".", "args": args}


def faster(p: Project, spec: TaskSpec, cs: list[dict]) -> dict[str, Any] | None:
    if not _check(cs, "speed"):
        return None
    ms = p.board().get(str(p.promoted), {}).get(spec.primary_suite(), {}).get("latency_ms") or 0
    why = (
        f"Takes {ms:.1f} ms per input on CPU, against a {spec.latency_budget_ms:g} ms budget. "
        "A smaller student learns from this model and keeps most of its accuracy."
    )
    return {"id": "faster", "title": "Make it faster", "why": why, "args": ["distill", p.name, "--run"]}
