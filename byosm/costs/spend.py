from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

from sqlalchemy import select

from ..db import session
from ..db.models import Job, JobEvent
from .prices import price

SUBCOMMANDS = ("baseline", "suite", "exp", "inputs", "system")
EVAL_ROLES = ("baseline", "eval", "card")
RIVAL_PREFIX = "rival-"
DONE = "done"


def target_of(args: list[str]) -> str | None:
    at = 2 if args and args[0] in SUBCOMMANDS else 1
    name = args[at] if len(args) > at else None
    return name if name and " " not in name and not name.startswith("--") else None


def is_eval(role: str) -> bool:
    return role in EVAL_ROLES or role.startswith(RIVAL_PREFIX)


def _usd(model: str, tokens_in: int, tokens_out: int) -> float | None:
    p = price(model)
    return None if p is None else tokens_in * p[0] + tokens_out * p[1]


def usage(ws: Path) -> dict[str, dict[str, Any]]:
    from ..app.store import JobStore

    known = JobStore(ws).targets()
    with session(ws) as s:
        owners = {j: known.get(j) or target_of(a) for j, a in s.execute(select(Job.id, Job.args))}
        fields = [JobEvent.data[k] for k in ("state", "role", "tokens_in", "tokens_out")]
        rows = s.execute(select(JobEvent.job, JobEvent.model, *fields).where(JobEvent.kind == "llm")).all()
    out: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"build": _bucket(), "evaluate": _bucket(), "rival": _bucket()}
    )
    for job, model, state, role, t_in, t_out in rows:
        name = owners.get(job)
        if not name or state != DONE:
            continue
        role, t_in, t_out = str(role), int(t_in or 0), int(t_out or 0)
        buckets = ["evaluate" if is_eval(role) else "build"] + (
            ["rival"] if role == "baseline" or role.startswith(RIVAL_PREFIX) else []
        )
        for b in buckets:
            _add(out[name][b], str(model), t_in, t_out)
    return dict(out)


def _bucket() -> dict[str, Any]:
    return {"calls": 0, "tokens_in": 0, "tokens_out": 0, "usd": 0.0, "unpriced": 0}


def _add(b: dict[str, Any], model: str, t_in: int, t_out: int) -> None:
    usd = _usd(model, t_in, t_out)
    b["calls"] += 1
    b["tokens_in"] += t_in
    b["tokens_out"] += t_out
    b["usd"] += usd or 0.0
    b["unpriced"] += usd is None
