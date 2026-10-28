from __future__ import annotations

import math
import time
from typing import Any

from ..evals.selection import sure_lead
from ..format import fmt
from ..improve.keys import proposal_key
from .standing import standing
from .state import Session

MIN_GAIN_WITHOUT_BOOTSTRAP = 0.002
PLATEAU_ROUNDS = 2


def _won_reason(st: dict) -> str:
    best = st["best_baseline"]
    vs_best = f" (best {best[0]} {best[1]:.4f})" if best else ""
    return f"won: {st['exp']} {st['ours']:.4f} beats every baseline{vs_best} and meets target {st['target']}"


def _budget_stop(s: Session, st: dict, max_experiments: int) -> str:
    if st["won"]:
        return _won_reason(st)
    n_exp = s.experiments_run()
    if n_exp >= max_experiments:
        return f"experiment budget reached ({n_exp}/{max_experiments})"
    est = s.mean_duration()
    if time.time() + est > s.deadline:
        return f"time budget: {(s.deadline - time.time()) / 3600:.1f}h left < ~{est / 3600:.1f}h per experiment"
    return ""


def _pick(fresh: list[dict], deadline: float) -> tuple[dict | None, str]:
    left_min = (deadline - time.time()) / 60
    fits = [pr for pr in fresh if pr["cost"]["minutes"] <= left_min]
    if not fits:
        cheapest = min(pr["cost"]["minutes"] for pr in fresh)
        return None, f"time budget: {left_min:.0f} min left, cheapest untried proposal needs ~{cheapest:.0f} min"
    return fits[0], ""


def _is_real_gain(kept: bool, cmp: dict | None, mine: float | None, before: float | None) -> bool:
    if not kept:
        return False
    if cmp:
        return sure_lead(cmp)
    return mine is not None and before is not None and mine - before > MIN_GAIN_WITHOUT_BOOTSTRAP


def _log_result(s: Session, st: dict, exp: str, pick: dict, fresh: list[dict], kept: bool, cmp: Any, mine: Any) -> None:
    before = st["ours"]
    p_better = f" (P(better)={cmp['p_better']:.2f})" if cmp else ""
    verdict = "kept (promoted)" if kept else "reverted"
    s.log(
        "experiment",
        exp,
        f"{pick['id']}: {st['suite']} calib {fmt(mine)} vs {fmt(before)}{p_better} -> {verdict}. Evidence: {pick['why']}",
        proposal=pick,
        delta=None if mine is None or before is None else round(mine - before, 4),
        p_better=cmp and cmp["p_better"],
        alternatives=[pr["id"] for pr in fresh if pr is not pick],
    )


def _experiment(s: Session, st: dict, pick: dict, fresh: list[dict]) -> bool | None:
    from .. import evals, improve
    from ..pipeline import build
    from ..train.gpu import waited_seconds

    exp = improve.apply(s.name, [pick["id"]])
    ts, queued = time.time(), waited_seconds()
    try:
        build(s.name, exp, s.echo, data_seconds=s.data_seconds())
    except Exception as e:
        s.log("experiment", exp, f"{pick['id']} failed: {type(e).__name__}: {str(e)[:200]}", proposal=pick)
        return None
    s.durations.append(time.time() - ts - (waited_seconds() - queued))
    kept = standing(s.name, s.margin)["exp"] == exp
    b = s.p.board()
    res = b.get(exp, {}).get(st["suite"])
    mine = evals.selection_value(res)
    cmp = evals.compare(s.base, res, b.get(st["exp"] or "", {}).get(st["suite"]))
    _log_result(s, st, exp, pick, fresh, kept, cmp, mine)
    return _is_real_gain(kept, cmp, mine, st["ours"])


def _applicable(s: Session, props: list[dict], tried: set[str]) -> list[dict]:
    fresh = []
    for pr in props:
        if proposal_key(pr) in tried:
            continue
        if math.isnan(pr["cost"]["minutes"]):
            tried.add(proposal_key(pr))
            s.log("skip", pr["id"], "its ops do not apply to the promoted spec", proposal=pr)
        else:
            fresh.append(pr)
    return fresh


def improve_until_stop(s: Session, max_experiments: int, use_llm: bool) -> str:
    from .. import improve
    from ..next_steps import tried_proposals

    tried = tried_proposals(s.p) | {
        proposal_key(d["proposal"]) for d in s.log.decisions if d["step"] in ("experiment", "skip") and "proposal" in d
    }
    stale = 0
    while True:
        st = standing(s.name, s.margin)
        reason = _budget_stop(s, st, max_experiments)
        if reason:
            return reason
        _, props = improve.propose(s.name, use_llm, s.echo)
        fresh = _applicable(s, props, tried)
        if not fresh:
            return "no untried proposals left"
        pick, reason = _pick(fresh, s.deadline)
        if pick is None:
            return reason
        tried.add(proposal_key(pick))
        real = _experiment(s, st, pick, fresh)
        stale = 0 if real else stale + 1
        if stale >= PLATEAU_ROUNDS:
            return f"plateau: {PLATEAU_ROUNDS} rounds without a significant gain or with a failed build"
