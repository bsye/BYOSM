from __future__ import annotations

from pathlib import Path

ARCHIVE = "archive"


def archived(ws: Path) -> set[str]:
    from .db.workspace import workspace_report

    return set((workspace_report(ws, ARCHIVE) or {}).get("names", []))


def set_archived(ws: Path, name: str, on: bool) -> set[str]:
    from .db.workspace import save_workspace_report

    names = archived(ws) - {name} | ({name} if on else set())
    save_workspace_report(ws, ARCHIVE, {"names": sorted(names)})
    return names
