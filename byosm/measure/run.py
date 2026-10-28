from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from .plan import columns, known, ordered, plan

RUN = "measure"
OPEN, DONE, FAILED = "open", "done", "failed"
TODO, PENDING, RUNNING = "todo", "pending", "running"
ACTIVE = ("queued", "running")
QUEUED_JOB = "queued"
QUEUE_TRIES = 5
EMPTY = {"version": 0, "started": None, "cells": []}


def saved_run(ws: Path) -> dict[str, Any]:
    from ..db.workspace import workspace_report

    return workspace_report(ws, RUN) or dict(EMPTY)


def save_run(ws: Path, run: dict[str, Any]) -> bool:
    from ..db import get_or_add, session
    from ..db.models import WorkspaceReport

    with session(ws) as s:
        row = get_or_add(s, WorkspaceReport, kind=RUN)
        if (row.data or EMPTY)["version"] != run["version"]:
            return False
        row.data = {**run, "version": run["version"] + 1}
    return True


def key(c: dict[str, Any]) -> tuple[str, str]:
    return c["specialist"], c["measurement"]


def merged(run: dict[str, Any], fresh: list[dict[str, Any]]) -> dict[str, Any]:
    busy = {key(c) for c in run["cells"] if c["state"] == OPEN}
    again = {key(f) for f in fresh}
    kept = [c for c in run["cells"] if c["state"] == OPEN or key(c) not in again]
    added = [{**f, "state": OPEN, "job": None, "jobs": [], "note": None} for f in fresh if key(f) not in busy]
    return {**run, "started": time.time(), "cells": ordered(kept + added)}


def queue(ws: Path, names: list[str] | None = None) -> list[dict[str, Any]]:
    fresh = plan(ws, names or None)
    for _ in range(QUEUE_TRIES):
        if not fresh or save_run(ws, merged(saved_run(ws), fresh)):
            return fresh
    raise RuntimeError("the measurement run kept changing while it was queued: try again")


def _live(c: dict[str, Any], jobs: dict[str, Any]) -> str:
    job = jobs.get(c["job"]) if c["job"] else None
    return PENDING if job is None or job.status == QUEUED_JOB else RUNNING


def _shown(c: dict[str, Any], status: str) -> dict[str, Any]:
    return {**{k: v for k, v in c.items() if k != "state"}, "status": status}


def _current(r: dict[str, Any] | None, f: dict[str, Any] | None) -> bool:
    return bool(r) and (r["state"] == OPEN or not f or (r["state"] == FAILED and r["model"] == f["model"]))


def cells(ran: list[dict[str, Any]], fresh: list[dict[str, Any]], jobs: dict[str, Any]) -> list[dict[str, Any]]:
    run, new = {key(c): c for c in ran}, {key(f): f for f in fresh}
    out = []
    for k in dict.fromkeys([*run, *new]):
        r, f = run.get(k), new.get(k)
        if _current(r, f):
            out.append(_shown(r, _live(r, jobs) if r["state"] == OPEN else r["state"]))
        else:
            out.append(_shown({**f, "job": None, "jobs": [], "note": None}, TODO))
    return out


def board(ws: Path, jobs: dict[str, Any], error: str | None = None) -> dict[str, Any]:
    run, names = saved_run(ws), known(ws)
    found = cells([c for c in run["cells"] if c["specialist"] in names], plan(ws), jobs)
    count = {s: sum(c["status"] == s for c in found) for s in (DONE, FAILED, TODO)}
    return {
        "measurements": columns(),
        "started": run["started"],
        "cells": found,
        "progress": {**count, "total": len(found)},
        "error": error,
    }
