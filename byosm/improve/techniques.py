from __future__ import annotations

from collections import Counter
from functools import cache
from pathlib import Path
from typing import Literal

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
    cost: float = 1.0
    prior: float = 0.5
    evidence: list[str] = Field(default_factory=list)
    status: Literal["available", "planned"] = "planned"

    def fits(self, spec: TaskSpec) -> bool:
        return self.status == AVAILABLE and (not self.modalities or spec.modality in self.modalities)


@cache
def techniques() -> tuple[Technique, ...]:
    return tuple(Technique.model_validate(t) for t in yaml.safe_load(CATALOGUE.read_text()))


def allowed_paths() -> list[str]:
    return [path for t in techniques() if t.status == AVAILABLE for path in t.paths]


def eligible(spec: TaskSpec) -> list[Technique]:
    return [t for t in techniques() if t.fits(spec)]


def family_of_path(path: str) -> str:
    from .paths import matches

    return next((t.id for t in techniques() for pattern in t.paths if matches(pattern, path)), OTHER)


def family(ops: list[dict]) -> str:
    found = Counter(family_of_path(o["path"]) for o in ops)
    return found.most_common(1)[0][0] if found else OTHER


def tried_counts(history: list[dict]) -> Counter:
    return Counter(h["family"] for h in history if h.get("family"))
