"""Distillation: shrink a (large) teacher experiment into a tiny student.

The student keeps the teacher's spec (task, labels, data, decision policy) with a small backbone, and trains on
alpha * gold-label loss + (1 - alpha) * T^2 KL(teacher || student), over the training inputs plus up to
`distill_unlabeled` extra unlabeled inputs that only the teacher labels.
"""
from __future__ import annotations

from typing import Any

from .config import Project
from .spec import TaskSpec

TINY_TEXT = "hf:google/bert_uncased_L-4_H-256_A-4"

# Default students per modality: (spec edits, short description)
STUDENTS: dict[str, tuple[dict[str, Any], str]] = {
    "text": ({"backbone": TINY_TEXT, "train.lr_backbone": 1e-4}, "4-layer BERT, 11M params"),
    "audio": ({"backbone": "tiny-crnn", "train.lr_backbone": 2e-3, "train.lr_head": 2e-3}, "causal conv+GRU, 0.1M params"),
    "image": ({"backbone": "tiny-cnn", "train.lr_backbone": 2e-3}, "small CNN, 0.4M params"),
    "document": ({"document.text_backbone": TINY_TEXT, "document.image_backbone": "tiny-cnn", "train.lr_backbone": 3e-4}, "tiny-BERT text + small CNN visual"),
}


def student_spec(teacher: TaskSpec, teacher_exp: str, overrides: dict[str, Any] | None = None) -> TaskSpec:
    from .improve import apply_ops

    edits, _ = STUDENTS[teacher.modality]
    edits = {**edits, **(overrides or {}), "train.distill_from": teacher_exp, "train.pretrain_steps": 0,
             "train.epochs": max(teacher.train.epochs, 12)}
    return apply_ops(teacher, [{"op": "set", "path": k, "value": v} for k, v in edits.items()])


def create(name: str, teacher_exp: str | None = None, overrides: dict[str, Any] | None = None, exp: str | None = None) -> str:
    import json

    p = Project(name)
    teacher_exp = teacher_exp or p.promoted
    if not teacher_exp:
        raise RuntimeError("no teacher: promote an experiment first")
    spec = student_spec(TaskSpec.load(p.exp_dir(teacher_exp) / "spec.yaml"), teacher_exp, overrides)
    exp = exp or f"distill-{teacher_exp}"
    d = p.exp_dir(exp)
    d.mkdir(parents=True, exist_ok=True)
    spec.save(d / "spec.yaml")
    (d / "proposal.json").write_text(json.dumps({"parent": teacher_exp, "applied": [{"id": "distill", "why": f"distill {teacher_exp} into {STUDENTS[spec.modality][1]}", "ops": []}]}, indent=2))
    return exp
