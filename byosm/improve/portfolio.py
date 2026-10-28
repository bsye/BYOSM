from __future__ import annotations

from ..project import Project, list_projects
from ..spec import TaskSpec
from .bandit import kind, outcomes
from .keys import proposal_key
from .techniques import transferable

PORTFOLIO_SIZE = 3
PORTFOLIO_PREFIX = "portfolio-"
PORTFOLIO = "portfolio"
BASE_PROBABILITY = 0.3
PER_WIN = 0.15
MAX_PROBABILITY = 0.9


def _wins(p: Project, spec: TaskSpec) -> dict[str, dict]:
    found: dict[str, dict] = {}
    for other in list_projects(p.ws):
        q = Project(other, p.ws)
        if other == p.name or kind(q.spec()) != kind(spec):
            continue
        for o in outcomes(q):
            ops = [op for op in o["ops"] if transferable(op)] if o["won"] else []
            if ops:
                found.setdefault(proposal_key({"ops": ops}), {"ops": ops, "won_on": set()})["won_on"].add(other)
    return found


def recipes(p: Project, spec: TaskSpec) -> list[dict]:
    best = sorted(_wins(p, spec).values(), key=lambda e: -len(e["won_on"]))[:PORTFOLIO_SIZE]
    out = []
    for i, e in enumerate(best, 1):
        won_on = sorted(e["won_on"])
        changes = ", ".join(f"{o['path']}={o['value']}" for o in e["ops"])
        why = f"Won a sure lead on {len(won_on)} other specialist(s) of this kind ({', '.join(won_on)}): {changes}."
        chance = min(MAX_PROBABILITY, BASE_PROBABILITY + PER_WIN * len(won_on))
        out.append(
            {"id": f"{PORTFOLIO_PREFIX}{i}", "why": why, "ops": e["ops"], "probability": chance, "source": PORTFOLIO}
        )
    return out
