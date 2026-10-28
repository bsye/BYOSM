from __future__ import annotations

from pathlib import Path
from typing import Any

from sqlalchemy import select

from .engine import get_or_add, session
from .models import WorkspaceReport


def workspace_report(ws: Path, kind: str) -> Any:
    with session(ws) as s:
        return s.scalar(select(WorkspaceReport.data).where(WorkspaceReport.kind == kind))


def save_workspace_report(ws: Path, kind: str, data: Any) -> None:
    with session(ws) as s:
        get_or_add(s, WorkspaceReport, kind=kind).data = data
