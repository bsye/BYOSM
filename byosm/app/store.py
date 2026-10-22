from __future__ import annotations

from pathlib import Path
from typing import Any

from sqlalchemy import func, select

from ..db import session
from ..db.models import Job, JobEvent, JobLine
from ..events import TARGET

JOB_FIELDS = ("id", "args", "status", "created", "started", "ended", "code", "pid")
USAGE_COLUMNS = (
    "job",
    "role",
    "model",
    "calls",
    "failed",
    "tokens_in",
    "tokens_out",
    "seconds",
    "last",
    "counted",
    "counted_seconds",
)
LLM_KIND = "llm"
ENDED_STATES = ("done", "failed")


def _usage_row(job: str, role: str | None, model: str | None) -> dict[str, Any]:
    return {**dict.fromkeys(USAGE_COLUMNS, 0), "job": job, "role": role, "model": model, "last": None}


def _tally(u: dict[str, Any], d: dict[str, Any]) -> None:
    u["calls" if d["state"] == "done" else "failed"] += 1
    u["tokens_in"] += d.get("tokens_in") or 0
    u["tokens_out"] += d.get("tokens_out") or 0
    u["seconds"] += d.get("seconds") or 0
    u["last"] = max(u["last"] or 0, d.get("at") or 0)
    if d.get("tokens_out") is not None:
        u["counted"] += 1
        u["counted_seconds"] += d.get("seconds") or 0


class JobStore:
    def __init__(self, ws: Path) -> None:
        self.ws = ws

    def save(self, job: dict[str, Any]) -> None:
        with session(self.ws) as s:
            s.merge(Job(**{k: job[k] for k in JOB_FIELDS}))

    def line(self, job_id: str, line: str) -> None:
        with session(self.ws) as s:
            s.add(JobLine(job=job_id, line=line))

    def event(self, job_id: str, data: dict[str, Any]) -> None:
        with session(self.ws) as s:
            s.add(JobEvent(job=job_id, kind=data.get("kind"), model=data.get("model"), data=data))

    def jobs(self) -> list[dict[str, Any]]:
        with session(self.ws) as s:
            return [{k: getattr(j, k) for k in JOB_FIELDS} for j in s.scalars(select(Job).order_by(Job.created))]

    def targets(self) -> dict[str, str]:
        with session(self.ws) as s:
            rows = s.execute(select(JobEvent.job, JobEvent.data).where(JobEvent.kind == TARGET)).all()
        return {job: d.get("name") for job, d in rows}

    def lines(self, job_id: str) -> list[str]:
        with session(self.ws) as s:
            return list(s.scalars(select(JobLine.line).where(JobLine.job == job_id).order_by(JobLine.id)))

    def seen(self, job_id: str) -> int:
        with session(self.ws) as s:
            n = [s.scalar(select(func.count()).select_from(t).where(t.job == job_id)) for t in (JobLine, JobEvent)]
        return sum(n)

    def events(self, job_id: str) -> list[dict[str, Any]]:
        with session(self.ws) as s:
            return list(s.scalars(select(JobEvent.data).where(JobEvent.job == job_id).order_by(JobEvent.id)))

    def llm_usage(self) -> list[dict[str, Any]]:
        with session(self.ws) as s:
            rows = s.execute(select(JobEvent.job, JobEvent.data).where(JobEvent.kind == LLM_KIND)).all()
        out: dict[tuple, dict[str, Any]] = {}
        for job, d in rows:
            if d.get("state") in ENDED_STATES:
                key = (job, d.get("role"), d.get("model"))
                _tally(out.setdefault(key, _usage_row(*key)), d)
        for u in out.values():
            u["seconds"], u["counted_seconds"] = round(u["seconds"], 1), round(u["counted_seconds"], 1)
        return sorted(out.values(), key=lambda u: -(u["last"] or 0))

    def jobs_using(self, model: str) -> set[str]:
        with session(self.ws) as s:
            q = select(JobEvent.job).where(JobEvent.kind == LLM_KIND, JobEvent.model == model).distinct()
            return set(s.scalars(q))
