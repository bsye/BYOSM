from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

from sqlalchemy import func, select

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
    role = JobEvent.data["role"].as_string()
    tokens = [func.coalesce(func.sum(JobEvent.data[k].as_integer()), 0) for k in ("tokens_in", "tokens_out")]
    done = JobEvent.data["state"].as_string() == DONE
    with session(ws) as s:
        owners = {j: known.get(j) or target_of(a) for j, a in s.execute(select(Job.id, Job.args))}
        found = select(JobEvent.job, JobEvent.model, role, func.count(), *tokens).where(JobEvent.kind == "llm", done)
        rows = s.execute(found.group_by(JobEvent.job, JobEvent.model, role)).all()
    out: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"build": _bucket(), "evaluate": _bucket(), "rival": _bucket()}
    )
    for job, model, role, calls, t_in, t_out in rows:
        name = owners.get(job)
        if not name:
            continue
        role = str(role)
        buckets = ["evaluate" if is_eval(role) else "build"] + (
            ["rival"] if role == "baseline" or role.startswith(RIVAL_PREFIX) else []
        )
        for b in buckets:
            _add(out[name][b], str(model), int(t_in), int(t_out), calls)
    return dict(out)


def _bucket() -> dict[str, Any]:
    return {"calls": 0, "tokens_in": 0, "tokens_out": 0, "usd": 0.0, "unpriced": 0}


def _add(b: dict[str, Any], model: str, t_in: int, t_out: int, calls: int = 1) -> None:
    usd = _usd(model, t_in, t_out)
    b["calls"] += calls
    b["tokens_in"] += t_in
    b["tokens_out"] += t_out
    b["usd"] += usd or 0.0
    b["unpriced"] += calls if usd is None else 0


def job_cost(ws: Path, job: str) -> dict[str, Any]:
    fields = [JobEvent.data[k] for k in ("state", "tokens_in", "tokens_out")]
    with session(ws) as s:
        rows = s.execute(select(JobEvent.model, *fields).where(JobEvent.job == job, JobEvent.kind == "llm")).all()
    out = _bucket()
    for model, state, t_in, t_out in rows:
        if state == DONE:
            _add(out, str(model), int(t_in or 0), int(t_out or 0))
    return out
