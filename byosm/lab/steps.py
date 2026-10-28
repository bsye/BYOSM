from __future__ import annotations

from pathlib import Path
from typing import Any

from .drafts import APPROVED, PENDING, drafts
from .owners import Owner

APPROVE_PREFIX = "approve-lab-"
RUN_PREFIX = "run-lab-"
GIST_CHARS = 200
ELLIPSIS = "..."


def _gist(text: str) -> str:
    first = text.partition(". ")[0].rstrip(".") + "."
    return first if len(first) <= GIST_CHARS else first[: GIST_CHARS - len(ELLIPSIS)].rstrip() + ELLIPSIS


def _not_run(d: dict[str, Any]) -> bool:
    return d.get("result") is None and not d.get("error") and not d.get("experiment")


def _approval(d: dict[str, Any]) -> dict[str, Any]:
    why = f"Waits for a person to read its code on the Lab tab. {_gist(d['hypothesis'])}"
    return {"id": APPROVE_PREFIX + d["id"], "title": f"Approve lab method {d['name']}", "why": why, "args": []} | {
        "approve": d["id"]
    }


def _run(name: str, d: dict[str, Any]) -> dict[str, Any]:
    why = (
        f"Refused last time: {d['refused']}" if d.get("refused") else f"Approved, not run yet. {_gist(d['hypothesis'])}"
    )
    return {
        "id": RUN_PREFIX + d["id"],
        "title": f"Run lab method {d['name']}",
        "why": why,
        "args": ["lab", "run", name, d["id"]],
    }


def lab_steps(owner: Owner) -> list[dict[str, Any]]:
    found = drafts(owner)
    pending = [_approval(d) for d in found if d["status"] == PENDING and not d["problems"]]
    ready = [_run(owner.name, d) for d in found if d["status"] == APPROVED and _not_run(d)]
    return pending + ready


def pending_approvals(ws: Path | None) -> list[dict[str, Any]]:
    from ..config import find_workspace
    from ..project import Project, list_projects
    from ..series.board import reports
    from .owners import Forecaster

    root = ws or find_workspace()
    owners: list[Owner] = [Project(n, root) for n in list_projects(root)]
    owners += [Forecaster(root, r["domain"]) for r in reports(root) if r["at"]]
    return [{"owner": o.name, **s} for o in owners for s in lab_steps(o) if s.get("approve")]
