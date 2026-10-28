from __future__ import annotations

from collections.abc import Callable
from typing import Any

from ..project import Project
from .drafts import APPROVED, draft, update

LAB_PATH = "train.lab_method"
LAB_PREFIX = "lab-"
ERROR_CHARS = 4000


def _experiment(p: Project, found: dict[str, Any]) -> str:
    from ..pipeline import next_exp

    parent = p.promoted
    spec = p.exp_spec(parent).model_copy(deep=True) if parent else p.spec()
    spec.train.lab_method = found["id"]
    exp = next_exp(p)
    op = {"op": "set", "path": LAB_PATH, "value": found["id"]}
    applied = [{"id": LAB_PREFIX + found["name"], "why": found["hypothesis"], "ops": [op]}]
    p.save_experiment(exp, spec=spec, proposal={"parent": parent, "applied": applied})
    return exp


def run(name: str, did: str, log: Callable[[str], None] = print) -> dict[str, Any]:
    from ..improve.bandit import VERDICT
    from ..pipeline import build

    p = Project(name)
    found = draft(p, did)
    if not found or found["status"] != APPROVED:
        raise PermissionError(f"lab method {did} is not approved")
    exp = _experiment(p, found)
    update(p, did, experiment=exp, error=None)
    log(f"[lab] {did} {found['name']} -> experiment {exp}")
    try:
        build(name, exp, log)
    except Exception as e:
        update(p, did, error=f"{type(e).__name__}: {e}"[:ERROR_CHARS])
        raise
    verdict = p.report(VERDICT + exp) or {}
    calib = (p.board().get(exp, {}).get(p.spec().primary_suite()) or {}).get("calib_objective")
    return update(p, did, result={"experiment": exp, "won": verdict.get("won"), "calib": calib})
