from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import yaml
from pydantic import BaseModel, Field
from sqlalchemy import select

from ..config import find_workspace
from ..db import get_or_add, session
from ..db.models import System, SystemReport
from ..spec import HFSource

if TYPE_CHECKING:
    from sqlalchemy.orm import Session

SYSTEM_FILE = "system.yaml"


class Node(BaseModel):
    specialist: str
    role: Literal["router", "tagger"]
    task: str
    source: HFSource
    train_splits: list[str] = Field(default_factory=lambda: ["train", "validation"])
    backbone: str | None = None


class Tool(BaseModel):
    when: str
    args: dict[str, str] = Field(default_factory=dict)
    call: str | None = None


class StreamCfg(BaseModel):
    max_extra_wrong: float = 1.0


class SystemSuite(BaseModel):
    name: str
    split: str = "test"
    max_items: int = 4000


class SystemSpec(BaseModel):
    name: str
    title: str = ""
    created: datetime | None = None
    description: str
    nodes: dict[str, Node]
    tools: dict[str, Tool] = Field(default_factory=dict)
    evals: list[SystemSuite] = Field(default_factory=list)
    baselines: list[str] = Field(default_factory=lambda: ["llm"])
    stream: StreamCfg = Field(default_factory=StreamCfg)

    def role(self, role: str) -> tuple[str, Node] | None:
        return next(((k, n) for k, n in self.nodes.items() if n.role == role), None)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(self.model_dump(mode="json", exclude_defaults=True), sort_keys=False))

    @classmethod
    def load(cls, path: Path) -> SystemSpec:
        return cls(**yaml.safe_load(path.read_text()))


class SystemProject:
    def __init__(self, name: str, ws: Path | None = None):
        self.ws = ws or find_workspace()
        self.name = name
        self.root = self.ws / "systems" / name

    @property
    def export_dir(self) -> Path:
        return self.root / "export"

    def _id(self, s: Session) -> int | None:
        return s.scalar(select(System.id).where(System.name == self.name))

    def exists(self) -> bool:
        with session(self.ws) as s:
            return self._id(s) is not None

    def spec(self) -> SystemSpec:
        with session(self.ws) as s:
            data = s.scalar(select(System.spec).where(System.name == self.name))
        if data is None:
            raise KeyError(f"no system {self.name!r}")
        return SystemSpec(**data)

    def save_spec(self, spec: SystemSpec) -> None:
        with session(self.ws) as s:
            get_or_add(s, System, name=self.name).spec = spec.model_dump(mode="json", exclude_defaults=True)

    def report(self, kind: str) -> Any:
        with session(self.ws) as s:
            return s.scalar(select(SystemReport.data).filter_by(system_id=self._id(s), kind=kind))

    def save_report(self, kind: str, data: Any) -> None:
        with session(self.ws) as s:
            get_or_add(s, SystemReport, system_id=self._id(s), kind=kind).data = data


def system_names(ws: Path) -> list[str]:
    with session(ws) as s:
        return sorted(s.scalars(select(System.name)))
