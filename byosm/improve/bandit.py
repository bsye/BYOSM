from __future__ import annotations

import random
from collections import defaultdict
from typing import Any

from ..health_score import CHECK_TITLES
from ..project import Project, list_projects
from ..spec import TaskSpec
from .registry import RULE
from .techniques import Technique, eligible, family, published

VERDICT = "verdict:"
PRIOR_STRENGTH = 2.0
POOL_WEIGHT = 0.5
MAX_FAILS = 3
ROUND_ONE_PRIOR = 0.3
RULE_BOOST = 1.5
BLOCKER_BOOST = 1.5

Tally = dict[str, list[float]]


def kind(spec: TaskSpec) -> tuple[str, str]:
    return spec.modality, spec.task_type


def outcomes(p: Project) -> list[dict[str, Any]]:
    verdicts = p.reports(VERDICT)
    out = []
    for exp in p.attempts():
        proposal = p.proposal(exp) or {}
        ops = [o for a in proposal.get("applied", []) for o in a.get("ops", [])]
        if not ops:
            continue
        v = verdicts.get(VERDICT + exp) or {"won": exp == p.promoted}
        out.append(
            {
                "experiment": exp,
                "parent": proposal.get("parent"),
                "family": family(ops),
                "won": bool(v["won"]),
                "gpu_hours": v.get("gpu_hours"),
                "ops": ops,
            }
        )
    return out


def _tally(rows: list[dict], weight: float, into: Tally) -> None:
    for r in rows:
        wins, tries = into.setdefault(r["family"], [0.0, 0.0])
        into[r["family"]] = [wins + weight * r["won"], tries + weight]


def family_stats(p: Project) -> tuple[Tally, Tally]:
    own: Tally = {}
    pooled: Tally = {}
    _tally(outcomes(p), 1.0, own)
    mine = kind(p.spec())
    for other in list_projects(p.ws):
        q = Project(other, p.ws)
        if other != p.name and kind(q.spec()) == mine:
            _tally(outcomes(q), POOL_WEIGHT, pooled)
    return own, pooled


def _dropped(t: Technique, own: Tally) -> bool:
    wins, tries = own.get(t.id, [0.0, 0.0])
    return not wins and tries >= MAX_FAILS


def _sample(t: Technique, own: Tally, pooled: Tally, rng: random.Random) -> float:
    wins = own.get(t.id, [0.0, 0.0])[0] + pooled.get(t.id, [0.0, 0.0])[0]
    tries = own.get(t.id, [0.0, 0.0])[1] + pooled.get(t.id, [0.0, 0.0])[1]
    a = 1 + t.prior * PRIOR_STRENGTH + wins
    b = 1 + (1 - t.prior) * PRIOR_STRENGTH + tries - wins
    return rng.betavariate(a, b) / max(t.cost, 0.1)


def _boost(t: Technique, boosted: set[str], failing: set[str]) -> float:
    return (RULE_BOOST if t.id in boosted else 1) * (BLOCKER_BOOST if failing & set(t.targets) else 1)


def family_order(
    p: Project, spec: TaskSpec, boosted: set[str] = frozenset(), seed: int = 0, failing: set[str] = frozenset()
) -> list[str]:
    own, pooled = family_stats(p)
    rng = random.Random(f"{p.name}:{len(p.attempts())}:{seed}")
    live = [t for t in eligible(spec) if not _dropped(t, own)] or eligible(spec)
    score = {t.id: _sample(t, own, pooled, rng) * _boost(t, boosted, failing) for t in live}
    fresh = {t.id for t in live if t.id not in own and t.prior >= ROUND_ONE_PRIOR}
    return sorted(score, key=lambda f: (f not in fresh, -score[f]))


def rank(
    p: Project, spec: TaskSpec, proposals: list[dict], seed: int = 0, failing: set[str] = frozenset()
) -> list[dict]:
    by_family: dict[str, list[dict]] = defaultdict(list)
    for pr in sorted(proposals, key=lambda pr: -(pr.get("probability") or 0)):
        by_family[pr["family"]].append(pr)
    ranked = family_order(p, spec, {pr["family"] for pr in proposals if pr.get("source") == RULE}, seed, failing)
    order = [f for f in ranked if f in by_family] + [f for f in by_family if f not in ranked]
    out: list[dict] = []
    while any(by_family[f] for f in order):
        out += [by_family[f].pop(0) for f in order if by_family[f]]
    return out


def search_summary(p: Project) -> dict[str, Any]:
    from .keys import proposal_key

    rows = outcomes(p)
    fams: dict[str, dict[str, Any]] = {}
    for r in rows:
        f = fams.setdefault(r["family"], {"family": r["family"], "tries": 0, "wins": 0, "gpu_hours": 0.0})
        f["tries"] += 1
        f["wins"] += r["won"]
        f["gpu_hours"] += r["gpu_hours"] or 0
    keys = [proposal_key({"ops": r["ops"]}) for r in rows]
    wins, hours = sum(r["won"] for r in rows), sum(r["gpu_hours"] or 0 for r in rows)
    return {
        "experiments": len(rows),
        "families": sorted(fams.values(), key=lambda f: -f["tries"]),
        "sure_leads": wins,
        "gpu_hours": round(hours, 3),
        "gpu_hours_per_sure_lead": round(hours / wins, 3) if wins else None,
        "repeats": len(keys) - len(set(keys)),
    }


def _seen(ws: Any) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    from .effects import measured

    tally: dict[str, dict[str, Any]] = {}
    found: list[dict[str, Any]] = []
    for name in list_projects(ws):
        p = Project(name, ws)
        rows = outcomes(p)
        found += measured(p, rows)
        for r in rows:
            t = tally.setdefault(r["family"], {"tries": 0, "wins": 0, "gpu_hours": 0.0, "specialists": set()})
            t["tries"] += 1
            t["wins"] += r["won"]
            t["gpu_hours"] += r["gpu_hours"] or 0
            t["specialists"].add(name)
    return tally, found


def catalogue_board(ws: Any = None) -> list[dict[str, Any]]:
    from .effects import family_effect
    from .techniques import techniques

    tally, found = _seen(ws)
    out = []
    for t in techniques():
        seen = tally.get(t.id, {"tries": 0, "wins": 0, "gpu_hours": 0.0, "specialists": set()})
        effect = family_effect([f for f in found if f["family"] == t.id])
        out.append(
            {
                **t.model_dump(),
                "target_titles": [CHECK_TITLES.get(c, c) for c in t.targets],
                "evidence_dates": [published(u) for u in t.evidence],
                **seen,
                "gpu_hours": round(seen["gpu_hours"], 3),
                "specialists": len(seen["specialists"]),
                "calib_effect": {**effect, "left_out": seen["tries"] - effect["measured"]},
            }
        )
    return out
