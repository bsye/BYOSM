from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ..data.store import TRAIN_INPUTS, Dataset, record
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
    from ..project import Project

    p = Project(spec.name)
    extra = _pool(spec, ds, train_rows)
    rows = train_rows + extra
    names = teachers(p, teacher_dir.name, spec.train.distill_seeds)
    log(f"  distill: {' + '.join(names)} label {len(train_rows)} train + {len(extra)} unlabeled inputs, averaged")
    soft = averaged([_probs(spec, p, teacher_dir.parent / n, data_dir, rows, dev) for n in names])
    extra_t = (
        [[IGNORE] * len(s) for s in soft[len(train_rows) :]] if spec.task_type == "frame" else [IGNORE] * len(extra)
    )
    return soft, mod.load_many(extra), extra_t


def teachers(p: Any, name: str, k: int) -> list[str]:
    from ..replicas import recipe_runs

    seeds = [e for e in recipe_runs(p, name) if e != name and p.meta(e)]
    return [name, *seeds][: max(1, k)]


def averaged(per_teacher: list[list[Any]]) -> list[Any]:
    return [np.mean(np.stack(row), axis=0) for row in zip(*per_teacher, strict=True)]


def _probs(spec: TaskSpec, p: Any, run_dir: Path, data_dir: Path, rows: list[dict], dev: str) -> list[Any]:
    from ..predictors import ExperimentPredictor

    name = run_dir.name
    tspec = p.exp_spec(name)
    teacher = ExperimentPredictor(
        tspec, p.meta(name), run_dir, name, device=dev if spec.task_type == "classify" else "cpu"
    )
    tmod = get_modality(tspec, str(data_dir))
    if spec.task_type == "frame":
        return [teacher.frame_probs(tmod.load(r)) for r in rows]
    return list(teacher.probs_batch(tmod.load_many(rows)))


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


PSEUDO = "pseudo"


def _pool(spec: TaskSpec, ds: Dataset, train_rows: list[dict]) -> list[dict]:
    seen = {str(r["input"]) for r in train_rows}
    unlabeled = [r for r in ds.unlabeled() if r.get("source") != TRAIN_INPUTS and str(r["input"]) not in seen]
    return unlabeled[: spec.train.distill_unlabeled]


def pseudo_labels(
    spec: TaskSpec, train_rows: list[dict], ds: Dataset, data_dir: Path, dev: str, log: Callable[[str], None]
) -> list[dict]:
    from ..project import Project

    p = Project(spec.name)
    pool = _pool(spec, ds, train_rows)
    if not p.promoted or not pool or spec.task_type != "classify":
        return []
    probs = np.asarray(_probs(spec, p, p.exp_dir(p.promoted), data_dir, pool, dev))
    sure = probs.max(1) >= spec.train.self_train
    names = spec.label_names
    out = [
        record(r["input"], names[int(k)], PSEUDO) for r, k, ok in zip(pool, probs.argmax(1), sure, strict=True) if ok
    ]
    log(
        f"  self-training: {p.promoted} labelled {len(out)} of {len(pool)} unlabelled inputs at >= {spec.train.self_train}"
    )
    return out
