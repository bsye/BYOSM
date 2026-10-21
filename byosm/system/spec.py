from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field

from ..config import find_workspace
from ..spec import HFSource

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


class SystemSuite(BaseModel):
    name: str
    split: str = "test"
    max_items: int = 4000


class SystemSpec(BaseModel):
    name: str
    description: str
    nodes: dict[str, Node]
    tools: dict[str, Tool] = Field(default_factory=dict)
    evals: list[SystemSuite] = Field(default_factory=list)
    baselines: list[str] = Field(default_factory=lambda: ["llm"])

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
    def spec_path(self) -> Path:
        return self.root / SYSTEM_FILE

    @property
    def export_dir(self) -> Path:
        return self.root / "export"

    def spec(self) -> SystemSpec:
        return SystemSpec.load(self.spec_path)
