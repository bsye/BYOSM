from __future__ import annotations

from collections.abc import Callable
from typing import Any

from ..project import Project
from .drafts import APPROVED, draft, update

MAX_REVISIONS = 2
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


def revisions(p: Project, did: str) -> int:
    depth, found = 0, draft(p, did)
    while found and found.get("parent"):
        depth, found = depth + 1, draft(p, found["parent"])
    return depth


def revise(name: str, found: dict[str, Any], why: str, log: Callable[[str], None]) -> dict[str, Any] | None:
    from .author import write

    if revisions(Project(name), found["id"]) >= MAX_REVISIONS:
        log(f"[lab] {found['name']} was revised {MAX_REVISIONS} times already; no new version")
        return None
    ask = f"Revise the method {found['name']} ({found['id']}): {why}. Keep what worked, change what the outcome points at."
    return write(name, ask, log, found.get("paper") or "", parent=found["id"])


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
        revise(name, found, f"its run failed with {type(e).__name__}: {str(e)[:ERROR_CHARS]}", log)
        raise
    verdict = p.report(VERDICT + exp) or {}
    calib = (p.board().get(exp, {}).get(p.spec().primary_suite()) or {}).get("calib_objective")
    result = update(p, did, result={"experiment": exp, "won": verdict.get("won"), "calib": calib})
    if not verdict.get("won"):
        revise(name, found, f"it reached calib {calib} and did not beat the promoted model", log)
    return result
