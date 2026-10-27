from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path

import yaml
from pydantic import BaseModel, Field

from .backbones import MINILM
from .sections import (
    Axis,
    DataPlan,
    Decision,
    DocumentCfg,
    EvalSuite,
    Label,
    Modality,
    Objective,
    TaskType,
    TrainCfg,
)

SYNTH_FIELDS = ("name", "summary", "input_description", "languages", "diversity_axes", "hard_cases")
DATA_RUN_SETTINGS = {"max_minutes", "local_source"}


class TaskSpec(BaseModel):
    name: str
    title: str = ""
    created: datetime | None = None
    description: str
    modality: Modality
    task_type: TaskType = "classify"
    summary: str = ""
    caveat: str = ""
    input_description: str = ""
    labels: list[Label]
    positive_label: str | None = None
    objective: Objective = Field(default_factory=Objective)
    languages: list[str] = Field(default_factory=lambda: ["en"])
    diversity_axes: list[Axis] = Field(default_factory=list)
    hard_cases: list[str] = Field(default_factory=list)
    backbone: str = MINILM
    latency_budget_ms: float = 10.0
    speed_tolerance: float = 0.01
    frame_hop_ms: float = 10.0
    data: DataPlan = Field(default_factory=DataPlan)
    train: TrainCfg = Field(default_factory=TrainCfg)
    decision: Decision = Field(default_factory=Decision)
    document: DocumentCfg = Field(default_factory=DocumentCfg)
    evals: list[EvalSuite] = Field(default_factory=lambda: [EvalSuite(name="builtin", primary=True)])
    baselines: list[str] = Field(default_factory=list)
    search: dict[str, list[str]] = Field(default_factory=dict)

    @property
    def label_names(self) -> list[str]:
        return [label.name for label in self.labels]

    @property
    def label_index(self) -> dict[str, int]:
        return {name: i for i, name in enumerate(self.label_names)}

    @property
    def pos_index(self) -> int:
        names = self.label_names
        return names.index(self.positive_label) if self.positive_label in names else len(self.labels) - 1

    def data_key(self) -> str:
        dump = self.model_dump(mode="json")
        blob = {
            "m": self.modality,
            "t": self.task_type,
            "l": self.label_names,
            "hop": self.frame_hop_ms,
            "d": self.data.model_dump(mode="json", exclude_defaults=True, exclude=DATA_RUN_SETTINGS),
            "desc": "",
        }
        if self.data.synthetic:
            blob |= {"l": dump["labels"], "desc": self.description, **{k: dump[k] for k in SYNTH_FIELDS}}
        return hashlib.sha1(json.dumps(blob, sort_keys=True).encode()).hexdigest()[:10]

    def synth_key(self) -> str:
        dump = self.model_dump(mode="json")
        blob = {"m": self.modality, "t": self.task_type, "l": dump["labels"], "desc": self.description}
        return hashlib.sha1(json.dumps(blob | {k: dump[k] for k in SYNTH_FIELDS}, sort_keys=True).encode()).hexdigest()[
            :10
        ]

    def primary_suite(self) -> str:
        public = (e.name for e in self.evals if e.kind != "generated")
        return next((e.name for e in self.evals if e.primary), next(public, "builtin"))

    def trains_on_test_split(self) -> bool:
        h = self.train.holdout_test
        prim = next((e for e in self.evals if e.name == self.primary_suite()), None)
        return h == "train" or (h == "auto" and prim is not None and prim.kind != "builtin")

    def save(self, path: Path) -> None:
        path.write_text(yaml.safe_dump(self.model_dump(mode="json"), sort_keys=False, allow_unicode=True, width=110))

    @classmethod
    def load(cls, path: Path) -> TaskSpec:
        return cls.model_validate(yaml.safe_load(path.read_text()))
