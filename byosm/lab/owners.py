from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..db.workspace import save_workspace_report, workspace_report, workspace_reports
from ..project import Project
from .contract import CLASSIFIER, FORECASTER

PREFIX = "forecaster:"


@dataclass(frozen=True)
class Forecaster:
    ws: Path
    name: str

    def _key(self, kind: str) -> str:
        return f"{PREFIX}{self.name}:{kind}"

    def report(self, kind: str) -> Any:
        return workspace_report(self.ws, self._key(kind))

    def reports(self, prefix: str) -> dict[str, Any]:
        return workspace_reports(self.ws, self._key(prefix))

    def save_report(self, kind: str, data: Any) -> None:
        save_workspace_report(self.ws, self._key(kind), data)


Owner = Project | Forecaster


def kind_of(owner: Owner) -> str:
    return FORECASTER if isinstance(owner, Forecaster) else CLASSIFIER


def owner(name: str, ws: Path | None = None) -> Owner:
    from ..series.board import report_of

    p = Project(name, ws)
    if p.exists():
        return p
    if (report_of(p.ws, name) or {}).get("at"):
        return Forecaster(p.ws, name)
    raise KeyError(f"no specialist or forecaster {name!r}")
