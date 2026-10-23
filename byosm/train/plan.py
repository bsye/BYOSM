from __future__ import annotations

from collections.abc import Callable
from typing import Any

from ..spec import TaskSpec
from .warm import CONTINUE_EPOCHS, continued

OVERFIT_LR_FACTOR = 0.25
OVERFIT_EPOCHS = 1
MORE_EPOCHS = 2
MAX_EPOCHS = 6
ROWS = {"ok": 300, "warn": 600, "fail": 1000}
FAILING = ("warn", "fail")
NOT_TRAINING = {
    "certainty": "the test score is uncertain: write a bigger test set (byosm testset) rather than train more",
    "speed": "it is too slow: distill it (byosm distill) rather than train more",
}


def _health(p: Any, parent: str) -> dict[str, dict[str, Any]]:
    from ..health import health

    if parent != p.promoted:
        return {}
    found = health(p, p.spec()) or {}
    return {c["id"]: c for c in found.get("checks", [])}


def _failing(checks: dict[str, dict[str, Any]], cid: str) -> dict[str, Any] | None:
    c = checks.get(cid)
    return c if c and c["status"] in FAILING else None


def _still_improving(meta: dict[str, Any]) -> bool:
    history, best = meta.get("history") or [], meta.get("best_epoch")
    return bool(history) and best is not None and best >= len(history) - 1


def recipe(p: Any, parent: str) -> tuple[TaskSpec, list[str], dict[str, dict[str, Any]]]:
    base, meta = p.exp_spec(parent), (p.experiment(parent) or {}).get("meta") or {}
    checks = _health(p, parent)
    reasons = [f"{NOT_TRAINING[cid]}" for cid in NOT_TRAINING if _failing(checks, cid)]
    if over := _failing(checks, "overfit"):
        spec = continued(base, parent, OVERFIT_EPOCHS)
        spec.train.lr_backbone = base.train.lr_backbone * OVERFIT_LR_FACTOR
        return spec, [*reasons, f"overfitting ({over['value']}): one epoch at a quarter of the learning rate"], checks
    behind = _failing(checks, "target") or _failing(checks, "rival")
    if _still_improving(meta) and (behind or not checks):
        epochs = min(MAX_EPOCHS, CONTINUE_EPOCHS + MORE_EPOCHS)
        return (
            continued(base, parent, epochs),
            [*reasons, f"dev was still improving at its last epoch: {epochs} epochs"],
            checks,
        )
    return continued(base, parent), [*reasons, f"{CONTINUE_EPOCHS} epochs at half the learning rate"], checks


def _aimed(errors: list[dict[str, Any]], checks: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    weak = _failing(checks, "weakest")
    label = (weak or {}).get("value", "").rpartition(" ")[0]
    return sorted(errors, key=lambda e: label not in (e.get("gold"), e.get("pred"))) if label else errors


def plan(p: Any, parent: str, exp: str, cfg: dict, log: Callable[[str], None]) -> TaskSpec:
    from ..pipeline import ensure_data, refine

    spec, reasons, checks = recipe(p, parent)
    errors = (p.experiment(parent) or {}).get("errors") or []
    data = (checks.get("data") or {}).get("status", "ok")
    spec.train.refine_examples = ROWS.get(data, ROWS["ok"])
    if _failing(checks, "robust"):
        reasons.append("it drops on new inputs: the rows it writes vary phrasing and context")
    if weak := _failing(checks, "weakest"):
        reasons.append(f"its weakest label comes first: {weak['value']}")
    if spec.modality == "text" and spec.task_type == "classify" and errors:
        found = refine(spec, cfg, _aimed(errors, checks), ensure_data(p, spec, cfg, log), exp, log)
        if found and found.get("added"):
            reasons.append(
                f"wrote {found['added']} rows aimed at the {len(errors)} dev errors {parent} made (data is {data})"
            )
    for r in reasons:
        log(f"  plan: {r}")
    return spec
