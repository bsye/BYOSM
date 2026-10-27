from __future__ import annotations

from pathlib import Path

ARCHIVE = "archive"


def archived(ws: Path) -> set[str]:
    from sqlalchemy import select

    from .db import session
    from .db.models import WorkspaceReport

    with session(ws) as s:
        data = s.scalar(select(WorkspaceReport.data).where(WorkspaceReport.kind == ARCHIVE))
    return set((data or {}).get("names", []))


def set_archived(ws: Path, name: str, on: bool) -> set[str]:
    from .db import get_or_add, session
    from .db.models import WorkspaceReport

    names = archived(ws) - {name} | ({name} if on else set())
    with session(ws) as s:
        get_or_add(s, WorkspaceReport, kind=ARCHIVE).data = {"names": sorted(names)}
    return names
