from __future__ import annotations

import json
from collections.abc import Callable

from ..config import Project
from ..spec import TaskSpec
from .analyst import llm_proposals
from .cost import estimate
from .diagnosis import diagnose
from .paths import apply_ops
from .registry import rule_proposals


def propose(name: str, use_llm: bool = True, log: Callable[[str], None] = print) -> tuple[dict, list[dict]]:
    dg = diagnose(name)
    props = rule_proposals(dg) + (llm_proposals(dg, log) if use_llm else [])
    for pr in props:
        pr["cost"] = estimate(dg, pr)
    p = Project(name)
    (p.root / "proposals.json").write_text(json.dumps({"from": dg["experiment"], "proposals": props}, indent=2))
    return dg, props


def apply(name: str, ids: list[str], exp: str | None = None) -> str:
    from ..pipeline import next_exp

    p = Project(name)
    saved = json.loads((p.root / "proposals.json").read_text())
    chosen = [pr for pr in saved["proposals"] if pr["id"] in ids]
    if len(chosen) != len(ids):
        raise KeyError(f"unknown proposal ids: {set(ids) - {pr['id'] for pr in chosen}}")
    parent = saved["from"]
    spec = apply_ops(TaskSpec.load(p.exp_dir(parent) / "spec.yaml"), [o for pr in chosen for o in pr["ops"]])
    exp = exp or next_exp(p)
    ed = p.exp_dir(exp)
    ed.mkdir(parents=True, exist_ok=True)
    spec.save(ed / "spec.yaml")
    (ed / "proposal.json").write_text(json.dumps({"parent": parent, "applied": chosen}, indent=2))
    return exp
