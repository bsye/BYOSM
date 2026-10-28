from __future__ import annotations

from collections import Counter
from functools import cache, reduce
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field

from ..spec import TaskSpec

CATALOGUE = Path(__file__).with_name("techniques.yaml")
AVAILABLE = "available"
OTHER = "other"


class Technique(BaseModel):
    id: str
    title: str
    effect: str
    paths: list[str] = Field(default_factory=list)
    modalities: list[str] = Field(default_factory=list)
    task_types: list[str] = Field(default_factory=list)
    requires: list[str] = Field(default_factory=list)
    cost: float = 1.0
    prior: float = 0.5
    evidence: list[str] = Field(default_factory=list)
    status: Literal["available", "built-in", "planned"] = "planned"
    transferable: bool = False

    def fits(self, modality: str, task_type: str | None = None) -> bool:
        kind = not self.modalities or modality in self.modalities
        task = not self.task_types or task_type is None or task_type in self.task_types
        return self.status == AVAILABLE and kind and task


def value_at(spec: TaskSpec, path: str) -> Any:
    return reduce(lambda part, key: getattr(part, key, None), path.split("."), spec)


def present(spec: TaskSpec, path: str) -> bool:
    return value_at(spec, path) is not None


@cache
def techniques() -> tuple[Technique, ...]:
    return tuple(Technique.model_validate(t) for t in yaml.safe_load(CATALOGUE.read_text()))


def allowed_paths() -> list[str]:
    return [path for t in techniques() if t.status == AVAILABLE for path in t.paths]


def eligible(spec: TaskSpec) -> list[Technique]:
    return [
        t for t in techniques() if t.fits(spec.modality, spec.task_type) and all(present(spec, r) for r in t.requires)
    ]


def family_of_op(op: dict) -> str:
    from .paths import claims

    return next((t.id for t in techniques() for pattern in t.paths if claims(pattern, op)), OTHER)


def family(ops: list[dict]) -> str:
    found = Counter(family_of_op(o) for o in ops)
    return found.most_common(1)[0][0] if found else OTHER


def transferable(op: dict) -> bool:
    fid = family_of_op(op)
    return any(t.id == fid and t.transferable for t in techniques())


def tried_counts(history: list[dict]) -> Counter:
    return Counter(h["family"] for h in history if h.get("family"))
