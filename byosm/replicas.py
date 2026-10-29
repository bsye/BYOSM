from __future__ import annotations

import math
from collections.abc import Callable
from statistics import mean, variance

from .evals.selection import clean_selection, selection_value
from .health_score import CHECK_TITLES, OK, WARN, check
from .project import Project

SEED_SEP = "-s"
SOUP_SUFFIX = "-soup"
SEED_STEP = "seed"
SEEDS_COMMAND = "seeds"
Z = 1.96


def is_soup(pid: str) -> bool:
    return pid.endswith(SOUP_SUFFIX) and len(pid) > len(SOUP_SUFFIX)


def recipe_of(pid: str) -> str | None:
    if is_soup(pid):
        return pid[: -len(SOUP_SUFFIX)]
    head, sep, tail = pid.rpartition(SEED_SEP)
    return head if sep and head and tail.isdigit() else None


def replica(exp: str, k: int) -> str:
    return f"{exp}{SEED_SEP}{k}"


def recipe_runs(p: Project, exp: str) -> list[str]:
    recipe = recipe_of(exp) or exp
    return [e for e in p.experiments() if not is_soup(e) and (e == recipe or recipe_of(e) == recipe)]


def clean_runs(p: Project, exp: str, suite: str) -> list[str]:
    board, spec = p.board(), p.spec()
    own = board.get(exp, {}).get(suite)
    if not own:
        return []
    same = [
        e for e in recipe_runs(p, exp) if board.get(e, {}).get(suite, {}).get("fingerprint") == own.get("fingerprint")
    ]
    return [e for e in same if clean_selection(p, spec, e)]


def runs(p: Project, exp: str, suite: str) -> list[dict]:
    board = p.board()
    found = clean_runs(p, exp, suite) or ([exp] if suite in board.get(exp, {}) else [])
    return [board[e][suite] for e in found]


def replicate(name: str, exp: str, n: int, log: Callable[[str], None] = print) -> list[str]:
    from .pipeline import build

    p = Project(name)
    suite = p.spec().primary_suite()
    spec = p.exp_spec(exp)
    recipe = recipe_of(exp) or exp
    made: list[str] = []
    for k in range(1, 2 * n):
        if len(clean_runs(p, exp, suite)) >= n:
            break
        rep = replica(recipe, k)
        if p.experiment(rep) is not None:
            continue
        seeded = spec.model_copy(deep=True)
        seeded.train.seed = spec.train.seed + k
        why = f"seed {seeded.train.seed} of {recipe}, to tell its lead from seed noise"
        p.save_experiment(
            rep, spec=seeded, proposal={"parent": exp, "applied": [{"id": SEED_STEP, "why": why, "ops": []}]}
        )
        log(f"[seeds] {rep}: {why}")
        build(name, rep, log, promote=False, suites=[suite])
        made.append(rep)
    return made


def score_rest(name: str, made: list[str], log: Callable[[str], None] = print) -> None:
    from .evals import run

    if made:
        log(f"[seeds] scoring {', '.join(made)} on the other suites")
        run(name, experiments=made, baselines=[], log=log)


def seed_sd(p: Project, suite: str) -> float | None:
    board = p.board()
    groups: dict[str, list[float]] = {}
    for e in p.experiments():
        value = selection_value(board.get(e, {}).get(suite))
        if value is not None and not is_soup(e) and clean_selection(p, p.spec(), e):
            groups.setdefault(recipe_of(e) or e, []).append(value)
    spread = [(len(v) - 1, variance(v)) for v in groups.values() if len(v) > 1]
    dof = sum(d for d, _ in spread)
    return math.sqrt(sum(d * v for d, v in spread) / dof) if dof else None


def beyond_seed_noise(mine: list[float], theirs: list[float], sd: float | None) -> bool:
    if sd is None or not mine or not theirs:
        return True
    return mean(mine) - mean(theirs) > Z * sd * math.sqrt(1 / len(mine) + 1 / len(theirs))


def seeds_check(p: Project, prim: str) -> dict:
    n, have = p.spec().finalist_seeds, len(clean_runs(p, p.promoted or "", prim))
    sd = seed_sd(p, prim)
    if have >= n and sd is not None:
        why = (
            "Spread of calib scores when the promoted recipe is trained again on other seeds: a smaller lead is noise."
        )
        return check("seeds", CHECK_TITLES["seeds"], 1, OK, f"sd {sd:.4f} over {have} seeds", why)
    why = f"The promoted model was trained once, so a lead smaller than its seed noise looks like a win. Trains it on {n} seeds."
    return check("seeds", CHECK_TITLES["seeds"], 1, WARN, f"{have} of {n} seeds", why, [SEEDS_COMMAND, p.name])
