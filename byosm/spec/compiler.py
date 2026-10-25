from __future__ import annotations

from datetime import datetime

from ..events import stage
from ..llm import get_llm
from ..project import free_name
from ..text import slug
from .backbones import BACKBONES, DEFAULT_BACKBONE
from .prompt import _SPEC_SCHEMA, _SPEC_SYSTEM
from .sections import Axis, DataPlan, Label, Objective, TrainCfg
from .task import TaskSpec

FROM_SCRATCH_TRAIN = {"pretrain_steps": 1500, "lr_backbone": 2e-3, "epochs": 30}
PRETRAINED_TRAIN = {"pretrain_steps": 200}
AUDIO_BATCH_SIZE = 64


def _prompt(description: str, modality: str | None) -> str:
    catalog = "\n".join(
        f"  [{m}] {b['id']}  ~{b['params_m']}M params, langs={b['langs']}: {b['note']}"
        for m, bs in BACKBONES.items()
        for b in bs
    )
    fixed = f"Modality is fixed to: {modality}\n\n" if modality else ""
    return f"Request:\n{description}\n\n" + fixed + f"Backbone catalog:\n{catalog}"


def _present(out: dict, fields: dict[str, str]) -> dict:
    return {field: out[key] for key, field in fields.items() if key in out}


def _given(out: dict, fields: dict[str, str]) -> dict:
    return {field: out[key] for key, field in fields.items() if out.get(key)}


def _backbone(mod: str, out: dict) -> str:
    known = {b["id"] for b in BACKBONES[mod]}
    if out.get("backbone") in known:
        return out.get("backbone")
    return DEFAULT_BACKBONE[mod]


def _objective(out: dict) -> Objective:
    return Objective(
        **_present(out, {"objective_metric": "metric", "objective_target": "target"}),
        **_given(out, {"objective_constraint": "constraint"}),
    )


def _data_plan(mod: str, out: dict) -> DataPlan:
    sizes = {field: int(n) for field, n in _given(out, {"train_size": "train_size", "test_size": "test_size"}).items()}
    return DataPlan(
        synthetic=bool(out.get("synthetic")) and mod == "text",
        **sizes,
        hf_queries=[*out.get("hf_queries", []), *_named(out.get("hf_candidates", []))],
    )


def _named(candidates: list[dict]) -> list[str]:
    names = (c["dataset"].split("/")[-1] for c in candidates if c.get("dataset"))
    return [" ".join(n.replace("_", " ").replace("-", " ").split()) for n in names]


def train_cfg(mod: str, backbone: str) -> TrainCfg:
    recipe = PRETRAINED_TRAIN if backbone.startswith("hf:") else FROM_SCRATCH_TRAIN
    batch = {"batch_size": AUDIO_BATCH_SIZE} if mod == "audio" else {}
    reading = {"truncation": "head_tail"} if mod == "text" else {}
    return TrainCfg(**recipe, **batch, **reading)


def _task_fields(out: dict) -> dict:
    present = ["title", "task_type", "summary", "caveat", "input_description", "hard_cases"]
    given = ["positive_label", "languages", "latency_budget_ms"]
    return {**_present(out, {k: k for k in present}), **_given(out, {k: k for k in given})}


@stage("Write the spec")
def compile_spec(name: str | None, description: str, modality: str | None = None, cfg: dict | None = None) -> TaskSpec:
    out = get_llm("spec", cfg).json(_SPEC_SYSTEM, _prompt(description, modality), _SPEC_SCHEMA)
    mod = modality or out["modality"]
    backbone = _backbone(mod, out)
    spec = TaskSpec(
        name=name or free_name(slug(out.get("name") or out.get("summary") or description)),
        created=datetime.now().astimezone(),
        description=description,
        modality=mod,
        labels=[Label(**label) for label in out["labels"]],
        objective=_objective(out),
        diversity_axes=[Axis(**a) for a in out.get("diversity_axes", [])],
        backbone=backbone,
        data=_data_plan(mod, out),
        search={"benchmark": out.get("benchmark_queries") or [], "baseline": out.get("baseline_queries") or []},
        train=train_cfg(mod, backbone),
        **_task_fields(out),
    )
    if spec.positive_label and spec.positive_label not in spec.label_names:
        spec.positive_label = None
    return spec
