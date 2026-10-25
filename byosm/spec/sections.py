from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from .backbones import DINOV2_SMALL, MINILM

Modality = Literal["text", "audio", "image", "document"]
TaskType = Literal["classify", "frame"]
Metric = Literal["f1", "macro_f1", "accuracy", "recall_at_precision", "precision_at_recall", "span_f1"]


HUB_PREFIX = "hf:"


class Label(BaseModel):
    name: str
    definition: str


class Axis(BaseModel):
    name: str
    values: list[str]


class Objective(BaseModel):
    metric: Metric = "macro_f1"
    target: float = 0.95
    constraint: float = 0.99


class HFSource(BaseModel):
    dataset: str
    role: Literal["labeled", "positive", "negative", "unlabeled"] = "labeled"
    format: Literal["rows", "markup", "segment_files"] = "rows"
    partition: Literal["train", "calib", "test", "all"] = "all"
    max_files: int = 40
    sampling: Literal["stream", "balanced"] = "stream"
    per_label: int = 100
    config: str | None = None
    split: str = "train"
    input_column: str | None = None
    label_column: str | None = None
    label_map: list[list[str]] = Field(default_factory=list)
    label_rules: list[list[str]] = Field(default_factory=list)
    label_default: str | None = None
    max_rows: int = 5000

    @field_validator("dataset")
    @classmethod
    def _bare_id(cls, v: str) -> str:
        return v.removeprefix(HUB_PREFIX)


class AudioCfg(BaseModel):
    sample_rate: int = 16000
    clip_seconds: float = 4.0
    snr_db: list[float] = Field(default_factory=lambda: [-5.0, 30.0])
    max_events: int = 3
    label_by_energy: bool = True
    fill_ms: int = 60


class DocumentCfg(BaseModel):
    route: Literal["text", "visual", "fused"] = "fused"
    text_backbone: str = MINILM
    image_backbone: str = DINOV2_SMALL
    image_pool: Literal["mean", "pooler"] = "mean"
    max_pages: int = 8
    max_len: int = 256
    image_size: int = 224
    dpi: int = 100
    ocr: Literal["auto", "always", "never"] = "auto"
    dim: int = 256
    modality_dropout: float = 0.2


class DataPlan(BaseModel):
    synthetic: bool = True
    train_size: int = 3000
    dev_size: int = 400
    test_size: int = 600
    per_call: int = 25
    max_minutes: float = 120.0
    judge: bool = True
    hf_queries: list[str] = Field(default_factory=list)
    hf_sources: list[HFSource] = Field(default_factory=list)
    local_dir: str | None = None
    teachers: list[str] = Field(default_factory=list)
    teacher_rows: int = 2000
    audio: AudioCfg = Field(default_factory=AudioCfg)


class Decision(BaseModel):
    kind: Literal["threshold", "hysteresis"] = "threshold"
    threshold: float | None = None
    deactivation_offset: float = 0.15
    min_on_ms: float = 50.0
    min_off_ms: float = 550.0
    smooth_alpha_per_32ms: float = 0.35


class EvalSuite(BaseModel):
    name: str
    kind: Literal["builtin", "hf", "local", "generated"] = "builtin"
    source: HFSource | None = None
    path: str | None = None
    max_items: int = 400
    primary: bool = False
    written_by: str | None = None
    judged_by: str | None = None

    @model_validator(mode="after")
    def _public_primary(self) -> EvalSuite:
        if self.primary and self.kind == "generated":
            raise ValueError(f"suite {self.name}: an LLM-written suite cannot be the primary benchmark")
        return self

    def label_space(self, names: list[str]) -> list[str]:
        targets = {b for _, b in self.source.label_map} if self.source else set()
        return [n for n in names if n in targets] or names


class TrainCfg(BaseModel):
    pretrain_steps: int = 300
    pretrain_lambda: float = 0.05
    epochs: int = 8
    batch_size: int = 32
    lr_backbone: float = 5e-5
    lr_head: float = 1e-3
    freeze_backbone: bool = False
    max_len: int = 128
    truncation: Literal["right", "head_tail"] = "right"
    refine_rounds: int = 2
    refine_examples: int = 300
    pool: Literal["mean", "attention"] = "mean"
    layer_decay_floor: float = 0.5
    probe_epochs: int = 0
    ema: bool = True
    seed: int = 0
    holdout_test: Literal["auto", "keep", "train"] = "auto"
    precision: Literal["auto", "fp32", "bf16", "fp16"] = "auto"
    init_from: str | None = None
    distill_from: str | None = None
    distill_alpha: float = 0.3
    distill_temperature: float = 2.0
    distill_unlabeled: int = 20000
    prefix_rate: float = 0.0
