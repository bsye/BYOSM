from __future__ import annotations

from collections.abc import Callable
from statistics import mean
from typing import Any

from . import evals
from .evals.scoring import fair_ms
from .project import Project
from .replicas import beyond_seed_noise, clean_runs, recipe_runs, replicate, runs, score_rest, seed_sd
from .soup import shipped
from .spec import TaskSpec

Log = Callable[[str], None]


def fixes_speed(spec: TaskSpec, mine: dict, cur: dict, mine_value: float, cur_value: float) -> bool:
    ms, was, budget = fair_ms(mine), fair_ms(cur), spec.latency_budget_ms
    in_budget = ms is not None and was is not None and ms <= budget < was
    return in_budget and mine_value >= cur_value - spec.speed_tolerance


def _values(rs: list[dict]) -> list[float]:
    return [v for r in rs if (v := evals.selection_value(r)) is not None]


def _seed_mean(rs: list[dict]) -> float:
    values = _values(rs)
    return mean(values) if values else -1


def _settle(p: Project, exp: str, prim: str, log: Log) -> list[str]:
    n, made = p.spec().finalist_seeds, []
    for e in (exp, p.promoted):
        if n > 1 and e and len(clean_runs(p, e, prim)) < n:
            made += replicate(p.name, e, n, log)
    return made


def _leads(p: Project, exp: str, prim: str) -> bool:
    board = p.board()
    a, b = board.get(exp, {}).get(prim), board.get(p.promoted or "", {}).get(prim)
    return bool(a and b and exp != p.promoted and _seed_mean([a]) > _seed_mean([b]))


def _record(p: Project, exp: str, won: bool, cmp: dict | None, runs_: tuple[list, list]) -> None:
    from .improve.bandit import VERDICT
    from .improve.effects import effects
    from .improve.techniques import family

    proposal = p.proposal(exp) or {}
    ops = [o for a in proposal.get("applied", []) for o in a.get("ops", [])]
    hours = sum(p.meta(e).get("gpu_hours") or 0 for e in recipe_runs(p, exp))
    verdict = {
        "won": won,
        "p_better": cmp["p_better"] if cmp else None,
        "seeds": [len(runs_[0]), len(runs_[1])],
        "family": family(ops) if ops else None,
        "gpu_hours": round(hours, 4),
    }
    p.save_report(VERDICT + exp, verdict)
    if won:
        from .lab.proven import note_win

        note_win(p, exp)
    if ops:
        effects(p, {exp: proposal.get("parent")})


def maybe_promote(p: Project, exp: str, log: Log = print) -> dict[str, Any]:
    base = p.spec()
    prim = base.primary_suite()
    if p.promoted and not evals.clean_selection(p, base, exp):
        log(
            f"  kept {p.promoted}: {exp} trained before calib was held out of dev, so its calib score is not a fair vote"
        )
        return p.board().get(exp, {})
    made = _settle(p, exp, prim, log) if _leads(p, exp, prim) else []
    board = p.board()
    a, b = board.get(exp, {}).get(prim), board.get(p.promoted or "", {}).get(prim) if p.promoted else None
    mine_runs, cur_runs = runs(p, exp, prim), runs(p, p.promoted, prim) if b else []
    mine, cur = _seed_mean(mine_runs), _seed_mean(cur_runs)
    cmp = evals.compare_runs(base, mine_runs, cur_runs) if cur_runs else None
    noise = seed_sd(p, prim)
    sure = evals.sure_lead(cmp) and beyond_seed_noise(_values(mine_runs), _values(cur_runs), noise)
    seeds = f" over {len(mine_runs)} vs {len(cur_runs)} seeds" if len(mine_runs) + len(cur_runs) > 2 else ""
    sig = f", P(better)={cmp['p_better']:.2f}" if cmp else ""
    sig += f", seed sd {noise:.4f}" if noise is not None else ""
    test = f"; test {a['objective']:.4f}" if a else ""
    won = (mine > cur and sure) or bool(a and b and fixes_speed(base, a, b, mine, cur))
    _record(p, exp, won, cmp, (mine_runs, cur_runs))
    if mine > cur and sure:
        why = f"{prim} calib {base.objective.metric} {mine:.4f} (was {cur:.4f}{seeds}{sig}){test}"
        p.promote(shipped(p, exp, mine_runs, log), why)
        log(f"  promoted {p.promoted}: {why}")
    elif a and b and fixes_speed(base, a, b, mine, cur):
        was = p.promoted
        why = (
            f"within {base.speed_tolerance} of {was} on {prim} calib ({mine:.4f} vs {cur:.4f}{seeds}) "
            f"and inside the {base.latency_budget_ms:g} ms budget that {was} misses{test}"
        )
        p.promote(exp, why)
        log(f"  promoted {exp}: {why}")
    elif mine > cur:
        log(f"  kept {p.promoted}: {exp}'s calib lead {mine:.4f} over {cur:.4f}{seeds} is not sure{sig}{test}")
    else:
        log(f"  kept {p.promoted}: {prim} calib {cur:.4f} >= {exp} {mine:.4f}{seeds}{sig}{test}")
    score_rest(p.name, made, log)
    return board.get(exp, {})
