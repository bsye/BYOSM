from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ..data.store import Dataset
from ..modalities import IGNORE, Modality, get_modality
from ..model import align_targets
from ..spec import TaskSpec


def _teacher_targets(
    spec: TaskSpec,
    mod: Modality,
    teacher_dir: Path,
    train_rows: list[dict],
    ds: Dataset,
    data_dir: Path,
    dev: str,
    log: Callable[[str], None],
) -> tuple[list[Any], list[Any], list[Any]]:
    from ..predictors import ExperimentPredictor
    from ..project import Project

    p, name = Project(spec.name), teacher_dir.name
    tspec = p.exp_spec(name)
    teacher_dev = dev if spec.task_type == "classify" else "cpu"
    teacher = ExperimentPredictor(tspec, p.meta(name), teacher_dir, name, device=teacher_dev)
    tmod = get_modality(tspec, str(data_dir))
    seen = {str(r["input"]) for r in train_rows}
    unlabeled = [r for r in ds.unlabeled() if r.get("source") != "train-inputs" and str(r["input"]) not in seen]
    extra = unlabeled[: spec.train.distill_unlabeled]
    rows = train_rows + extra
    log(
        f"  distill: teacher {teacher_dir.name} ({teacher.params / 1e6:.1f}M) "
        f"labels {len(train_rows)} train + {len(extra)} unlabeled inputs"
    )
    if spec.task_type == "frame":
        soft = [teacher.frame_probs(tmod.load(r)) for r in rows]
        extra_t = [[IGNORE] * len(s) for s in soft[len(train_rows) :]]
    else:
        soft = list(teacher.probs_batch(tmod.load_many(rows)))
        extra_t = [IGNORE] * len(extra)
    return soft, mod.load_many(extra), extra_t


def soft_batch(
    spec: TaskSpec, soft: list[Any], b: list[int], logits: torch.Tensor, mask: torch.Tensor, dev: str
) -> tuple[torch.Tensor, torch.Tensor]:
    if spec.task_type != "frame":
        return torch.as_tensor(np.stack([soft[i] for i in b])).to(dev), torch.ones(len(b), device=dev)
    tp = torch.nn.utils.rnn.pad_sequence([torch.as_tensor(soft[i]) for i in b], batch_first=True).to(dev)
    lens = torch.tensor([len(soft[i]) for i in b], device=dev)
    valid = (torch.arange(tp.shape[1], device=dev)[None] < lens[:, None]).float()
    return align_targets(tp, logits.shape[1]), align_targets(valid[..., None], logits.shape[1])[..., 0] * mask.float()


def row_soft_targets(spec: TaskSpec, rows: list[dict]) -> list[np.ndarray] | None:
    if not any(r.get("meta", {}).get("soft") for r in rows):
        return None
    eye = np.eye(len(spec.labels), dtype=np.float32)
    return [np.asarray(r["meta"].get("soft") or eye[spec.label_index[r["label"]]], np.float32) for r in rows]
