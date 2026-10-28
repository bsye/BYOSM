from __future__ import annotations

from collections.abc import Callable
from typing import Any

from ..project import Project
from .contract import FORECASTER
from .drafts import APPROVED, draft, update
from .owners import Forecaster, Owner, kind_of, owner

MAX_REVISIONS = 2
LAB_PATH = "train.lab_method"
LAB_PREFIX = "lab-"
ERROR_CHARS = 4000
Log = Callable[[str], None]


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


def revisions(found_owner: Owner, did: str) -> int:
    depth, found = 0, draft(found_owner, did)
    while found and found.get("parent"):
        depth, found = depth + 1, draft(found_owner, found["parent"])
    return depth


def revise(name: str, found: dict[str, Any], why: str, log: Log) -> dict[str, Any] | None:
    from .author import write

    if revisions(owner(name), found["id"]) >= MAX_REVISIONS:
        log(f"[lab] {found['name']} was revised {MAX_REVISIONS} times already; no new version")
        return None
    ask = f"Revise the method {found['name']} ({found['id']}): {why}. Keep what worked, change what the outcome points at."
    return write(name, ask, log, found.get("paper") or "", parent=found["id"])


def _classifier(p: Project, found: dict[str, Any], log: Log) -> dict[str, Any]:
    from ..improve.bandit import VERDICT
    from ..pipeline import build

    exp = _experiment(p, found)
    update(p, found["id"], experiment=exp, error=None)
    log(f"[lab] {found['id']} {found['name']} -> experiment {exp}")
    build(p.name, exp, log)
    verdict = p.report(VERDICT + exp) or {}
    calib = (p.board().get(exp, {}).get(p.spec().primary_suite()) or {}).get("calib_objective")
    return {"experiment": exp, "won": verdict.get("won"), "calib": calib}


def _forecaster(f: Forecaster, found: dict[str, Any], log: Log) -> dict[str, Any]:
    from ..series.run import try_lab_method

    update(f, found["id"], error=None)
    log(f"[lab] {found['id']} {found['name']} -> a new version of {f.name}")
    report = try_lab_method(f.ws, f.name, found["id"], log)
    newest = report["versions"][-1]
    update(f, found["id"], experiment=newest["name"])
    return {"experiment": newest["name"], "won": report["promoted"] == newest["name"], "calib": newest["calib"]}


def run(name: str, did: str, log: Log = print) -> dict[str, Any]:
    found_owner = owner(name)
    found = draft(found_owner, did)
    if not found or found["status"] != APPROVED:
        raise PermissionError(f"lab method {did} is not approved")
    trained = _forecaster if kind_of(found_owner) == FORECASTER else _classifier
    try:
        result = trained(found_owner, found, log)
    except Exception as e:
        update(found_owner, did, error=f"{type(e).__name__}: {e}"[:ERROR_CHARS])
        revise(name, found, f"its run failed with {type(e).__name__}: {str(e)[:ERROR_CHARS]}", log)
        raise
    updated = update(found_owner, did, result=result)
    if not result["won"]:
        revise(name, found, f"it reached calib {result['calib']} and did not beat the promoted model", log)
    return updated
