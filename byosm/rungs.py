from __future__ import annotations

import math
from collections.abc import Callable

from .deadline import BudgetSpentError
from .evals.selection import selection_value
from .project import Project
from .spec import TaskSpec

RUNG_SUFFIX = "-r1"
RUNG_STEP = "rung"


def rung_name(exp: str) -> str:
    return f"{exp}{RUNG_SUFFIX}"


def is_rung(exp: str) -> bool:
    return exp.endswith(RUNG_SUFFIX)


def auxiliary(exp: str) -> bool:
    from .ablation import is_arm
    from .curve import is_curve_run

    return is_rung(exp) or is_arm(exp) or is_curve_run(exp)


def short(spec: TaskSpec) -> TaskSpec:
    s = spec.model_copy(deep=True)
    s.train.epochs = max(1, math.ceil(spec.train.epochs / max(1, spec.screen_eta)))
    s.train.refine_rounds = 0
    return s


def _rung_score(p: Project, exp: str, log: Callable[[str], None]) -> float | None:
    from .pipeline import build

    rung = rung_name(exp)
    if not p.meta(rung):
        why = f"{exp} trained for 1/{p.spec().screen_eta} of its epochs, to screen it before a full training"
        proposal = {"parent": exp, "applied": [{"id": RUNG_STEP, "why": why, "ops": []}]}
        p.save_experiment(rung, spec=short(p.exp_spec(exp)), proposal=proposal)
        try:
            build(p.name, rung, log, promote=False, suites=[p.spec().primary_suite()])
        except BudgetSpentError:
            raise
        except Exception as e:
            log(f"[screen] {rung} failed: {type(e).__name__}: {str(e)[:160]}")
            return None
    return selection_value(p.board().get(rung, {}).get(p.spec().primary_suite()))


def screen(name: str, exps: list[str], log: Callable[[str], None] = print) -> list[str]:
    p = Project(name)
    eta = p.spec().screen_eta
    if eta <= 1 or len(exps) < 2:
        return exps
    scores = {e: s for e in exps if (s := _rung_score(p, e, log)) is not None}
    ranked = sorted(scores, key=lambda e: -scores[e])
    kept = ranked[: math.ceil(len(exps) / eta)]
    shown = ", ".join(f"{e} {scores[e]:.4f}" for e in ranked)
    log(f"[screen] calib after a short run: {shown}; full training for {', '.join(kept) or 'none'}")
    return kept
