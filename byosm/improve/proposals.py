from __future__ import annotations

from collections.abc import Callable

from pydantic import ValidationError

from ..data.store import ShingleIndex
from ..events import stage
from ..project import Project
from ..spec import TaskSpec
from .analyst import LLM, llm_proposals
from .bandit import rank
from .cost import estimate
from .diagnosis import diagnose
from .keys import proposal_key
from .paths import apply_ops
from .portfolio import recipes
from .registry import rule_proposals, unique
from .techniques import family

PROPOSALS = "proposals"
NOVELTY_SIMILARITY = 0.9


def _valid(spec: TaskSpec, proposal: dict, log: Callable[[str], None]) -> bool:
    try:
        changed = apply_ops(spec, proposal["ops"]) != spec
    except (ValidationError, KeyError, IndexError, TypeError) as e:
        log(f"  (dropped {proposal['id']}: its edits make an invalid spec: {str(e).splitlines()[0][:120]})")
        return False
    if not changed:
        log(f"  (dropped {proposal['id']}: its edits leave the spec as it is)")
    return changed


def _novel(p: Project, props: list[dict], log: Callable[[str], None]) -> list[dict]:
    seen = ShingleIndex([{"input": a["why"]} for e in p.attempts() for a in _applied(p, e) if a.get("why")])
    out = []
    for pr in props:
        hit = seen.nearest(pr["why"]) if pr.get("source") == LLM and pr["why"] else None
        if hit and hit[1] >= NOVELTY_SIMILARITY:
            log(f"  (dropped {pr['id']}: its idea restates an earlier experiment)")
            continue
        seen.add(pr["why"])
        out.append(pr)
    return out


@stage("Find what to try next")
def propose(name: str, use_llm: bool = True, log: Callable[[str], None] = print) -> tuple[dict, list[dict]]:
    dg = diagnose(name)
    p = Project(name)
    base = p.exp_spec(dg["experiment"]) if dg.get("experiment") else p.spec()
    found = recipes(p, base) + rule_proposals(dg) + (llm_proposals(dg, log) if use_llm else [])
    props = _novel(p, [pr for pr in unique(found) if _valid(base, pr, log)], log)
    for pr in props:
        pr["cost"] = estimate(dg, pr)
        pr["family"] = family(pr["ops"])
    props = rank(p, base, props)
    p.save_report(PROPOSALS, {"from": dg["experiment"], "proposals": props})
    return dg, props


class AlreadyTriedError(ValueError):
    pass


def _applied(p: Project, exp: str) -> list[dict]:
    return (p.proposal(exp) or {}).get("applied", [])


def _same_attempt(p: Project, keys: set[str]) -> str | None:
    return next((e for e in p.attempts() if {proposal_key(a) for a in _applied(p, e)} == keys), None)


def _explicit_sources(parent: TaskSpec, spec: TaskSpec) -> TaskSpec:
    if parent.data.hf_sources and not spec.data.hf_sources:
        spec.data.hf_queries = []
    return spec


def apply(name: str, ids: list[str], exp: str | None = None, force: bool = False) -> str:
    from ..pipeline import next_exp

    p = Project(name)
    saved = p.report(PROPOSALS) or {"proposals": []}
    chosen = [pr for pr in saved["proposals"] if pr["id"] in ids]
    if len(chosen) != len(ids):
        raise KeyError(f"unknown proposal ids: {set(ids) - {pr['id'] for pr in chosen}}")
    if (same := _same_attempt(p, {proposal_key(pr) for pr in chosen})) and not force:
        if same not in p.experiments():
            return same
        raise AlreadyTriedError(f"{', '.join(ids)} already ran as {same}")
    parent = saved["from"]
    base = p.exp_spec(parent)
    spec = _explicit_sources(base, apply_ops(base, [o for pr in chosen for o in pr["ops"]]))
    exp = exp or next_exp(p)
    p.save_experiment(exp, spec=spec, proposal={"parent": parent, "applied": chosen})
    return exp
