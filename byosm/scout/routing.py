from __future__ import annotations

from pathlib import Path
from typing import Any

from ..db.workspace import save_workspace_report
from .proposals import SCOUT, state

SERIES = "series"
LAB_KINDS = ("text", "audio", "image", "document", SERIES)
MAX_TRIES = 2


def lab_fits(proposal: dict[str, Any]) -> bool:
    return bool(set(proposal.get("kinds") or []) & set(LAB_KINDS))


def _specialists(ws: Path | None, kinds: set[str]) -> list[tuple[int, str]]:
    from ..archive import archived
    from ..health import health
    from ..project import Project, list_projects

    found = []
    for name in sorted(set(list_projects(ws)) - archived(ws)):
        p = Project(name, ws)
        spec = p.spec()
        if p.promoted and spec.modality in kinds:
            found.append(((health(p, spec) or {}).get("score", 0), name))
    return found


def _forecasters(ws: Path | None, kinds: set[str]) -> list[tuple[int, str]]:
    from ..series.board import promoted_reports
    from ..series.health import health

    if SERIES not in kinds:
        return []
    return [((health(r) or {}).get("score", 0), r["domain"]) for r in promoted_reports(ws)]


def chosen(ws: Path | None, proposal: dict[str, Any]) -> list[str]:
    kinds = set(proposal.get("kinds") or []) & set(LAB_KINDS)
    return [name for _, name in sorted(_specialists(ws, kinds) + _forecasters(ws, kinds))[:MAX_TRIES]]


def ask(proposal: dict[str, Any]) -> str:
    return f'Implement "{proposal["title"]}" from {proposal["paper"]}: {proposal.get("how") or proposal["effect"]}'


def mark_sent(ws: Path | None, pid: str, names: list[str]) -> dict[str, Any]:
    st = state(ws)
    found = next(p for p in st["proposals"] if p["id"] == pid)
    found["sent"] = sorted({*found.get("sent", []), *names})
    save_workspace_report(ws, SCOUT, st)
    return found
