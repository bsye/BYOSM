"""TaskSpec: the single source of truth for a specialist. Compiled from natural language, editable as YAML."""
from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional

import yaml
from pydantic import BaseModel, Field

from .llm import get_llm

Modality = Literal["text", "audio", "image"]
TaskType = Literal["classify", "frame"]  # frame = one label per timestep/token (VAD, tagging)
Metric = Literal["f1", "macro_f1", "accuracy", "recall_at_precision", "precision_at_recall"]

# Backbones the spec compiler may choose from. Any `hf:<repo>` id also works if you edit spec.yaml.
BACKBONES: dict[str, list[dict]] = {
    "text": [
        {"id": "hf:google/bert_uncased_L-4_H-256_A-4", "params_m": 11, "langs": "en", "note": "tiny BERT, ~1ms on CPU"},
        {"id": "hf:sentence-transformers/all-MiniLM-L6-v2", "params_m": 22, "langs": "en", "note": "strong small English encoder"},
        {"id": "hf:sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", "params_m": 118, "langs": "50+", "note": "small multilingual (most params are embeddings)"},
        {"id": "hf:distilbert-base-multilingual-cased", "params_m": 135, "langs": "104", "note": "distilled mBERT"},
        {"id": "hf:google-bert/bert-base-multilingual-cased", "params_m": 178, "langs": "104", "note": "mBERT"},
        {"id": "hf:answerdotai/ModernBERT-base", "params_m": 150, "langs": "en+code", "note": "long context (8k), English"},
    ],
    "audio": [
        {"id": "tiny-crnn", "params_m": 0.2, "langs": "any", "note": "log-mel conv+GRU, causal, 10ms frames, streaming-ready; trained from scratch (JEPA pretraining helps a lot)"},
        {"id": "hf:facebook/wav2vec2-base", "params_m": 95, "langs": "en-pretrained", "note": "20ms frames, strong speech features, heavy"},
    ],
    "image": [
        {"id": "tiny-cnn", "params_m": 0.4, "langs": "-", "note": "small CNN from scratch, <1ms"},
        {"id": "hf:facebook/dinov2-small", "params_m": 22, "langs": "-", "note": "DINOv2 ViT-S/14, excellent frozen features"},
    ],
}


class Label(BaseModel):
    name: str
    definition: str


class Axis(BaseModel):
    name: str
    values: list[str]


class Objective(BaseModel):
    metric: Metric = "macro_f1"
    target: float = 0.95
    # For *_at_* metrics: the constraint, e.g. precision >= 0.99 while maximizing recall.
    constraint: float = 0.99


class HFSource(BaseModel):
    dataset: str
    role: Literal["labeled", "positive", "negative", "unlabeled"] = "labeled"
    # rows: a `datasets` table.  segment_files: a repo of audio files + JSON sidecars with onset/offset lists
    # (e.g. AVA-Speech); frame labels come from the segments.
    format: Literal["rows", "segment_files"] = "rows"
    # Stable hash partition of a source's items: train | calib | test | all. The same dataset can feed training
    # (train) and an eval suite (calib/test) with zero overlap.
    partition: Literal["train", "calib", "test", "all"] = "all"
    max_files: int = 40
    config: Optional[str] = None
    split: str = "train"
    input_column: Optional[str] = None
    label_column: Optional[str] = None
    # [[source_value, our_label], ...]; source values not listed are dropped.
    label_map: list[list[str]] = Field(default_factory=list)
    max_rows: int = 5000


class AudioCfg(BaseModel):
    sample_rate: int = 16000
    clip_seconds: float = 4.0
    snr_db: list[float] = Field(default_factory=lambda: [-5.0, 30.0])
    max_events: int = 3
    # Frame labels come from the positive source's own energy envelope (so pauses inside speech are 0).
    label_by_energy: bool = True
    # Gaps shorter than this inside an event stay positive. ~60ms = word-level labels, ~300ms = utterance-level.
    fill_ms: int = 60


class DataPlan(BaseModel):
    synthetic: bool = True  # LLM-written examples (text)
    train_size: int = 3000
    dev_size: int = 400  # adversarial, generated like test; used for calibration + refinement
    test_size: int = 600  # adversarial, held out, never seen by the refine loop
    per_call: int = 25
    judge: bool = True  # second-opinion relabel; disagreements are dropped
    hf_queries: list[str] = Field(default_factory=list)
    hf_sources: list[HFSource] = Field(default_factory=list)
    local_dir: Optional[str] = None  # <dir>/<label>/* files
    audio: AudioCfg = Field(default_factory=AudioCfg)


class Decision(BaseModel):
    """How probabilities become decisions at deployment. Calibrated on each suite's calib split if values are None."""

    kind: Literal["threshold", "hysteresis"] = "threshold"
    threshold: Optional[float] = None
    # hysteresis (frame tasks), LiveKit/Silero-style: smooth, enter at `threshold`, leave below threshold - offset,
    # require min_on_ms to start and min_off_ms of silence to end.
    deactivation_offset: float = 0.15
    min_on_ms: float = 50.0
    min_off_ms: float = 550.0
    smooth_alpha_per_32ms: float = 0.35


class EvalSuite(BaseModel):
    """A named, frozen evaluation set. `builtin` = this specialist's own held-out test (dev used for calibration)."""

    name: str
    kind: Literal["builtin", "hf", "local"] = "builtin"
    source: Optional[HFSource] = None  # kind=hf (rows or segment_files); partitions calib/test are used
    path: Optional[str] = None  # kind=local: jsonl with input/label (+ split: calib|test)
    max_items: int = 400
    primary: bool = False  # the suite that decides which experiment gets promoted


class TrainCfg(BaseModel):
    pretrain_steps: int = 300  # LeJEPA domain adaptation on unlabeled task inputs; 0 disables
    pretrain_lambda: float = 0.05  # SIGReg weight
    epochs: int = 8
    batch_size: int = 32
    lr_backbone: float = 5e-5
    lr_head: float = 1e-3
    freeze_backbone: bool = False
    max_len: int = 128  # text tokens
    refine_rounds: int = 2
    refine_examples: int = 300


class TaskSpec(BaseModel):
    name: str
    description: str
    modality: Modality
    task_type: TaskType = "classify"
    summary: str = ""
    input_description: str = ""
    labels: list[Label]
    positive_label: Optional[str] = None
    objective: Objective = Field(default_factory=Objective)
    languages: list[str] = Field(default_factory=lambda: ["en"])
    diversity_axes: list[Axis] = Field(default_factory=list)
    hard_cases: list[str] = Field(default_factory=list)
    backbone: str = "hf:sentence-transformers/all-MiniLM-L6-v2"
    latency_budget_ms: float = 10.0
    frame_hop_ms: float = 10.0
    data: DataPlan = Field(default_factory=DataPlan)
    train: TrainCfg = Field(default_factory=TrainCfg)
    decision: Decision = Field(default_factory=Decision)
    evals: list[EvalSuite] = Field(default_factory=lambda: [EvalSuite(name="builtin", primary=True)])
    baselines: list[str] = Field(default_factory=list)  # e.g. ["silero"], ["llm"]

    @property
    def label_names(self) -> list[str]:
        return [l.name for l in self.labels]

    @property
    def pos_index(self) -> int:
        return self.label_names.index(self.positive_label) if self.positive_label in self.label_names else len(self.labels) - 1

    def data_key(self) -> str:
        """Content hash of everything that determines the dataset; experiments with equal keys share data."""
        import hashlib
        import json

        # Definitions and the task description only shape the data when an LLM writes it.
        labels = self.model_dump(mode="json")["labels"] if self.data.synthetic else self.label_names
        blob = {"m": self.modality, "t": self.task_type, "l": labels, "hop": self.frame_hop_ms,
                "d": self.data.model_dump(mode="json"), "desc": self.description if self.data.synthetic else ""}
        return hashlib.sha1(json.dumps(blob, sort_keys=True).encode()).hexdigest()[:10]

    def primary_suite(self) -> str:
        return next((e.name for e in self.evals if e.primary), self.evals[0].name if self.evals else "builtin")

    def save(self, path: Path) -> None:
        path.write_text(yaml.safe_dump(self.model_dump(mode="json"), sort_keys=False, allow_unicode=True, width=110))

    @classmethod
    def load(cls, path: Path) -> "TaskSpec":
        return cls.model_validate(yaml.safe_load(path.read_text()))


_SPEC_SCHEMA = {
    "type": "object",
    "properties": {
        "modality": {"type": "string", "enum": ["text", "audio", "image"]},
        "task_type": {"type": "string", "enum": ["classify", "frame"]},
        "summary": {"type": "string"},
        "input_description": {"type": "string"},
        "labels": {"type": "array", "items": {"type": "object", "properties": {"name": {"type": "string"}, "definition": {"type": "string"}}}},
        "positive_label": {"type": "string"},
        "objective_metric": {"type": "string", "enum": ["f1", "macro_f1", "accuracy", "recall_at_precision", "precision_at_recall"]},
        "objective_target": {"type": "number"},
        "objective_constraint": {"type": "number"},
        "languages": {"type": "array", "items": {"type": "string"}},
        "diversity_axes": {"type": "array", "items": {"type": "object", "properties": {"name": {"type": "string"}, "values": {"type": "array", "items": {"type": "string"}}}}},
        "hard_cases": {"type": "array", "items": {"type": "string"}},
        "backbone": {"type": "string"},
        "latency_budget_ms": {"type": "number"},
        "synthetic": {"type": "boolean"},
        "hf_queries": {"type": "array", "items": {"type": "string"}},
        "hf_candidates": {"type": "array", "items": {"type": "object", "properties": {"dataset": {"type": "string"}, "role": {"type": "string", "enum": ["labeled", "positive", "negative", "unlabeled"]}}}},
        "train_size": {"type": "integer"},
        "test_size": {"type": "integer"},
    },
}

_SPEC_SYSTEM = """You are the task compiler of byojl, a framework that trains tiny single-purpose perception models \
("System 1" specialists, in the spirit of JEPA/LeJEPA encoders with a small task head). Turn a natural-language request \
into a precise, unambiguous task specification.

Rules:
- Labels must be mutually exclusive and collectively exhaustive for the input distribution. Add a catch-all label \
(e.g. "other"/"none") when real inputs can fall outside the named classes. Definitions must settle borderline cases explicitly.
- task_type "frame" means one label per timestep/token (e.g. voice activity per 10ms frame, token tagging). Otherwise "classify".
- positive_label: for binary detection tasks, the label whose precision/recall matters most; "" otherwise.
- objective: choose what the deployer actually cares about. For triggers/detectors where false alarms are costly use \
recall_at_precision with constraint e.g. 0.99 (maximize recall subject to precision >= constraint). target is the value of \
the metric we aim to reach on the held-out adversarial test set.
- diversity_axes: 3-6 orthogonal axes of variation of real inputs (domain, register, length, speaker, noise, language...). \
Each with 3-8 concrete values. These drive synthetic data diversity.
- hard_cases: 8-20 specific, realistic confusers and edge cases that a naive model gets wrong.
- backbone: pick the SMALLEST backbone from the catalog that can plausibly reach the target in the latency budget. \
Multilingual needs -> multilingual backbone.
- synthetic: true if an LLM can write realistic inputs (text). For audio/image it must be false; rely on Hugging Face datasets.
- hf_queries: Hugging Face Hub search terms to find real data (labeled sets, or for audio/image: positive sources, \
negative/background sources, hard negatives). hf_candidates: specific dataset repo ids you are confident exist \
(role: labeled = already has our labels after mapping; positive/negative = raw material for mixing; unlabeled = in-domain inputs \
for self-supervised pretraining).
- train_size / test_size: enough for a very precise model (typically 2000-8000 train, 400-1000 test)."""


def compile_spec(name: str, description: str, modality: str | None = None, cfg: dict | None = None) -> TaskSpec:
    catalog = "\n".join(f"  [{m}] {b['id']}  ~{b['params_m']}M params, langs={b['langs']}: {b['note']}" for m, bs in BACKBONES.items() for b in bs)
    prompt = f"Request:\n{description}\n\n" + (f"Modality is fixed to: {modality}\n\n" if modality else "") + f"Backbone catalog:\n{catalog}"
    out = get_llm("spec", cfg).json(_SPEC_SYSTEM, prompt, _SPEC_SCHEMA)

    mod = modality or out["modality"]
    known = {b["id"] for b in BACKBONES[mod]}
    backbone = out.get("backbone") if out.get("backbone") in known else BACKBONES[mod][0 if mod != "text" else 1]["id"]
    from_scratch = not backbone.startswith("hf:")
    task_type = out.get("task_type", "classify")
    spec = TaskSpec(
        name=name,
        description=description,
        modality=mod,
        task_type=task_type,
        summary=out.get("summary", ""),
        input_description=out.get("input_description", ""),
        labels=[Label(**l) for l in out["labels"]],
        positive_label=out.get("positive_label") or None,
        objective=Objective(metric=out.get("objective_metric", "macro_f1"), target=out.get("objective_target", 0.95), constraint=out.get("objective_constraint") or 0.99),
        languages=out.get("languages") or ["en"],
        diversity_axes=[Axis(**a) for a in out.get("diversity_axes", [])],
        hard_cases=out.get("hard_cases", []),
        backbone=backbone,
        latency_budget_ms=out.get("latency_budget_ms") or 10.0,
        data=DataPlan(
            synthetic=bool(out.get("synthetic")) and mod == "text",
            train_size=int(out.get("train_size") or 3000),
            test_size=int(out.get("test_size") or 600),
            hf_queries=out.get("hf_queries", []),
            hf_sources=[HFSource(dataset=c["dataset"], role=c["role"]) for c in out.get("hf_candidates", [])],
        ),
        train=TrainCfg(
            pretrain_steps=1500 if from_scratch else 200,
            lr_backbone=2e-3 if from_scratch else 5e-5,
            epochs=30 if from_scratch else 8,
            batch_size=64 if mod == "audio" else 32,
        ),
    )
    if spec.positive_label and spec.positive_label not in spec.label_names:
        spec.positive_label = None
    return spec
