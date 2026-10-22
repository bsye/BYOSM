from __future__ import annotations

from collections.abc import Callable

from pydantic import ValidationError

from ..events import stage
from ..project import Project
from ..spec import TaskSpec
from .analyst import llm_proposals
from .cost import estimate
from .diagnosis import diagnose
from .paths import apply_ops
from .registry import rule_proposals, unique

PROPOSALS = "proposals"


def _valid(spec: TaskSpec, proposal: dict, log: Callable[[str], None]) -> bool:
    try:
        apply_ops(spec, proposal["ops"])
    except (ValidationError, KeyError, IndexError, TypeError) as e:
        log(f"  (dropped {proposal['id']}: its edits make an invalid spec: {str(e).splitlines()[0][:120]})")
        return False
    return True


@stage("Find what to try next")
def propose(name: str, use_llm: bool = True, log: Callable[[str], None] = print) -> tuple[dict, list[dict]]:
    dg = diagnose(name)
    p = Project(name)
    base = p.exp_spec(dg["experiment"]) if dg.get("experiment") else p.spec()
    found = unique(rule_proposals(dg) + (llm_proposals(dg, log) if use_llm else []))
    props = [pr for pr in found if _valid(base, pr, log)]
    for pr in props:
        pr["cost"] = estimate(dg, pr)
    p.save_report(PROPOSALS, {"from": dg["experiment"], "proposals": props})
    return dg, props


def apply(name: str, ids: list[str], exp: str | None = None) -> str:
    from ..pipeline import next_exp

    p = Project(name)
    saved = p.report(PROPOSALS) or {"proposals": []}
    chosen = [pr for pr in saved["proposals"] if pr["id"] in ids]
    if len(chosen) != len(ids):
        raise KeyError(f"unknown proposal ids: {set(ids) - {pr['id'] for pr in chosen}}")
    parent = saved["from"]
    spec = apply_ops(p.exp_spec(parent), [o for pr in chosen for o in pr["ops"]])
    exp = exp or next_exp(p)
    p.save_experiment(exp, spec=spec, proposal={"parent": parent, "applied": chosen})
    return exp
