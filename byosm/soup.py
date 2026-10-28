from __future__ import annotations

from collections.abc import Callable
from statistics import mean

from .evals.selection import selection_value
from .project import WEIGHTS, Project
from .replicas import SOUP_SUFFIX, clean_runs, recipe_of

SOUP_STEP = "soup"
MIN_MEMBERS = 2


def soup_name(exp: str) -> str:
    return f"{recipe_of(exp) or exp}{SOUP_SUFFIX}"


def _average(p: Project, members: list[str]) -> dict:
    import torch

    states = [torch.load(p.exp_dir(m) / WEIGHTS, map_location="cpu") for m in members]
    first = states[0]
    return {
        k: (sum(s[k].float() for s in states) / len(states)).to(v.dtype) if v.is_floating_point() else v
        for k, v in first.items()
    }


def make_soup(name: str, exp: str, log: Callable[[str], None] = print) -> str | None:
    import torch

    from . import evals

    p = Project(name)
    members = clean_runs(p, exp, p.spec().primary_suite())
    if len(members) < MIN_MEMBERS:
        return None
    out = soup_name(exp)
    p.exp_dir(out).mkdir(parents=True, exist_ok=True)
    torch.save(_average(p, members), p.exp_dir(out) / WEIGHTS)
    why = f"average of the weights of {', '.join(members)}"
    meta = {**p.meta(members[0]), "soup_of": members, "calib_held": True, "gpu_hours": 0.0}
    proposal = {"parent": exp, "applied": [{"id": SOUP_STEP, "why": why, "ops": []}]}
    p.save_experiment(out, spec=p.exp_spec(members[0]), meta=meta, proposal=proposal)
    log(f"[soup] {out}: {why}")
    evals.run(name, experiments=[out], log=log)
    return out


def shipped(p: Project, exp: str, runs: list[dict], log: Callable[[str], None] = print) -> str:
    soup = make_soup(p.name, exp, log) if len(runs) >= MIN_MEMBERS else None
    found = selection_value(p.board().get(soup or "", {}).get(p.spec().primary_suite()))
    seeds = [v for r in runs if (v := selection_value(r)) is not None]
    if soup and found is not None and seeds and found >= mean(seeds):
        log(f"  shipping {soup}: calib {found:.4f} >= the {len(seeds)}-seed mean {mean(seeds):.4f}")
        return soup
    return exp
