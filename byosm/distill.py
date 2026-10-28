from __future__ import annotations

from typing import Any

from .project import Project
from .spec import TaskSpec
from .spec.sections import FINETUNE

STUDENT_PREFIX = "distill-"
TINY_TEXT = "hf:google/bert_uncased_L-4_H-256_A-4"
TINY_VISION = "hf:facebook/deit-tiny-patch16-224"

STUDENTS: dict[str, tuple[dict[str, Any], str]] = {
    "text": ({"backbone": TINY_TEXT, "train.lr_backbone": 1e-4}, "4-layer BERT, 11M params"),
    "audio": (
        {"backbone": "tiny-crnn", "train.lr_backbone": 2e-3, "train.lr_head": 2e-3},
        "causal conv+GRU, 0.1M params",
    ),
    "image": ({"backbone": TINY_VISION, "train.lr_backbone": 1e-4}, "DeiT-tiny, 5.7M params"),
    "document": (
        {"document.text_backbone": TINY_TEXT, "document.image_backbone": TINY_VISION, "train.lr_backbone": 1e-4},
        "tiny-BERT text + DeiT-tiny visual, ~17M params",
    ),
}


def student_spec(teacher: TaskSpec, teacher_exp: str, overrides: dict[str, Any] | None = None) -> TaskSpec:
    from .improve import apply_ops

    edits, _ = STUDENTS[teacher.modality]
    edits = {
        **edits,
        **(overrides or {}),
        "train.method": FINETUNE,
        "train.distill_from": teacher_exp,
        "train.pretrain_steps": 0,
        "train.epochs": max(teacher.train.epochs, 12),
    }
    return apply_ops(teacher, [{"op": "set", "path": k, "value": v} for k, v in edits.items()])


def student_name(teacher: str) -> str:
    return f"{STUDENT_PREFIX}{teacher}"


def create(
    name: str, teacher_exp: str | None = None, overrides: dict[str, Any] | None = None, exp: str | None = None
) -> str:
    p = Project(name)
    teacher_exp = teacher_exp or p.promoted
    if not teacher_exp:
        raise RuntimeError("no teacher: promote an experiment first")
    spec = student_spec(p.exp_spec(teacher_exp), teacher_exp, overrides)
    exp = exp or student_name(teacher_exp)
    why = f"distill {teacher_exp} into {STUDENTS[spec.modality][1]}"
    proposal = {"parent": teacher_exp, "applied": [{"id": "distill", "why": why, "ops": []}]}
    p.save_experiment(exp, spec=spec, proposal=proposal)
    return exp
