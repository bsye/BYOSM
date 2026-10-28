from __future__ import annotations

import json
import math
from collections.abc import Callable
from typing import Any

from pydantic import ValidationError

from ..events import stage
from ..project import Project
from ..spec import TaskSpec
from .analyst import llm_proposals
from .cost import estimate
from .diagnosis import diagnose
from .paths import apply_ops
from .registry import rule_proposals, unique
from .techniques import family

PROPOSALS = "proposals"
LOG_BINS_PER_DECADE = 4
EXACT_PATHS = {"train.seed"}


def _bucket(v: Any) -> Any:
    if isinstance(v, str):
        return " ".join(v.lower().split())
    if isinstance(v, bool) or not isinstance(v, int | float) or v == 0:
        return v
    return math.copysign(10 ** (round(math.log10(abs(v)) * LOG_BINS_PER_DECADE) / LOG_BINS_PER_DECADE), v)


def normalized(ops: list[dict]) -> list[dict]:
    out = [{**o, "value": o["value"] if o["path"] in EXACT_PATHS else _bucket(o.get("value"))} for o in ops]
    return sorted(out, key=lambda o: json.dumps(o, sort_keys=True, default=str))


def proposal_key(pr: dict) -> str:
    return json.dumps(normalized(pr["ops"]), sort_keys=True, default=str)


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
        pr["family"] = family(pr["ops"])
    p.save_report(PROPOSALS, {"from": dg["experiment"], "proposals": props})
    return dg, props


class AlreadyTriedError(ValueError):
    pass


def _same_attempt(p: Project, keys: set[str]) -> str | None:
    return next(
        (e for e in p.attempts() if {proposal_key(a) for a in (p.proposal(e) or {}).get("applied", [])} == keys), None
    )


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
