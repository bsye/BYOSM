from __future__ import annotations

from collections.abc import Callable
from statistics import mean
from typing import Any

from . import evals
from .evals.scoring import fair_ms
from .evals.selection import selected, selection_mean, selection_value
from .project import Project
from .replicas import beyond_seed_noise, clean_runs_all, recipe_runs, replicate, score_rest, seed_sd, selection_runs
from .soup import shipped
from .spec import TaskSpec

Log = Callable[[str], None]


def fixes_speed(spec: TaskSpec, mine: dict, cur: dict, mine_value: float, cur_value: float) -> bool:
    ms, was, budget = fair_ms(mine), fair_ms(cur), spec.latency_budget_ms
    in_budget = ms is not None and was is not None and ms <= budget < was
    return in_budget and mine_value >= cur_value - spec.speed_tolerance


def _values(board: dict, ids: list[str], suites: list[str]) -> list[float]:
    return [v for e in ids if (v := selection_mean(board.get(e, {}), suites)) is not None]


def _seed_mean(values: list[float]) -> float:
    return mean(values) if values else -1


def _settle(p: Project, exp: str, suites: list[str], log: Log) -> list[str]:
    n, made = p.spec().finalist_seeds, []
    for e in (exp, p.promoted):
        if n > 1 and e and len(clean_runs_all(p, e, suites)) < n:
            made += replicate(p.name, e, n, log)
    return made


def _leads(p: Project, exp: str, suites: list[str]) -> bool:
    board = p.board()
    a, b = selection_mean(board.get(exp, {}), suites), selection_mean(board.get(p.promoted or "", {}), suites)
    return a is not None and b is not None and exp != p.promoted and a > b


def _by_suite(board: dict, ids: list[str], suites: list[str]) -> dict[str, list[dict]]:
    return {s: [board[e][s] for e in ids] for s in suites}


def _deltas(board: dict, mine: list[str], cur: list[str], suites: list[str]) -> str:
    if len(suites) < 2 or not mine or not cur:
        return ""
    gap = {
        s: mean(selection_value(board[e][s]) for e in mine) - mean(selection_value(board[e][s]) for e in cur)
        for s in suites
    }
    return ", " + ", ".join(f"{s} {d:+.4f}" for s, d in gap.items())


def _vote(p: Project, base: TaskSpec, exp: str, suites: list[str]) -> dict[str, Any]:
    board, prim = p.board(), base.primary_suite()
    a = board.get(exp, {}).get(prim)
    b = board.get(p.promoted or "", {}).get(prim) if p.promoted else None
    mine_ids, cur_ids = selection_runs(p, exp, suites), selection_runs(p, p.promoted, suites) if b else []
    mine_v, cur_v = _values(board, mine_ids, suites), _values(board, cur_ids, suites)
    cmp = (
        evals.compare_suites(base, _by_suite(board, mine_ids, suites), _by_suite(board, cur_ids, suites))
        if cur_ids
        else None
    )
    noise = seed_sd(p, suites)
    paired = cmp is not None or not cur_ids
    sig = f", P(better)={cmp['p_better']:.2f}" if cmp else ""
    sig += f", seed sd {noise:.4f}" if noise is not None else ""
    return {
        "board": board,
        "a": a,
        "b": b,
        "ids": (mine_ids, cur_ids),
        "cmp": cmp,
        "mine": _seed_mean(mine_v),
        "cur": _seed_mean(cur_v),
        "sure": paired and evals.sure_lead(cmp) and beyond_seed_noise(mine_v, cur_v, noise),
        "where": f"{prim} calib" if suites == [prim] else f"calib mean over {', '.join(suites)}",
        "seeds": f" over {len(mine_ids)} vs {len(cur_ids)} seeds" if len(mine_ids) + len(cur_ids) > 2 else "",
        "sig": sig + _deltas(board, mine_ids, cur_ids, suites),
        "test": f"; test {a['objective']:.4f}" if a else "",
    }


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
    if p.promoted and not evals.clean_selection(p, base, exp):
        log(
            f"  kept {p.promoted}: {exp} trained before calib was held out of dev, so its calib score is not a fair vote"
        )
        return p.board().get(exp, {})
    suites = selected(base, p.board().get(p.promoted or exp, {}))
    made = _settle(p, exp, suites, log) if _leads(p, exp, suites) else []
    v = _vote(p, base, exp, suites)
    a, b, mine, cur, seeds, sig, test = v["a"], v["b"], v["mine"], v["cur"], v["seeds"], v["sig"], v["test"]
    faster = bool(a and b and fixes_speed(base, a, b, mine, cur))
    _record(p, exp, (mine > cur and v["sure"]) or faster, v["cmp"], v["ids"])
    if mine > cur and v["sure"]:
        why = f"{v['where']} {base.objective.metric} {mine:.4f} (was {cur:.4f}{seeds}{sig}){test}"
        p.promote(shipped(p, exp, v["ids"][0], suites, log), why)
        log(f"  promoted {p.promoted}: {why}")
    elif faster:
        was = p.promoted
        why = (
            f"within {base.speed_tolerance} of {was} on {v['where']} ({mine:.4f} vs {cur:.4f}{seeds}) "
            f"and inside the {base.latency_budget_ms:g} ms budget that {was} misses{test}"
        )
        p.promote(exp, why)
        log(f"  promoted {exp}: {why}")
    elif mine > cur:
        log(
            f"  kept {p.promoted}: {exp}'s lead on {v['where']} {mine:.4f} over {cur:.4f}{seeds} is not sure{sig}{test}"
        )
    else:
        log(f"  kept {p.promoted}: {v['where']} {cur:.4f} >= {exp} {mine:.4f}{seeds}{sig}{test}")
    score_rest(p.name, made, log)
    return v["board"].get(exp, {})
