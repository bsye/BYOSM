from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .. import jepa
from ..data.store import Dataset
from ..modalities import Modality, get_modality
from ..model import SpecialistModel
from ..spec import TaskSpec

PRETRAIN_MAX_LEN = 256


def pretrain_backbone(
    spec: TaskSpec,
    model: SpecialistModel,
    mod: Modality,
    ds: Dataset,
    train_raw: list[Any],
    dev: str,
    log: Callable[[str], None],
    seed: int,
) -> list[dict]:
    tc = spec.train
    if tc.pretrain_steps <= 0:
        return []
    unl = mod.load_many(ds.unlabeled()) or train_raw
    log(f"  LeJEPA: {tc.pretrain_steps} steps on {len(unl)} unlabeled inputs (lambda={tc.pretrain_lambda})")
    views = _short_views(spec, mod)
    return jepa.pretrain(
        model, views, unl, tc.pretrain_steps, tc.batch_size, tc.lr_backbone, tc.pretrain_lambda, dev, log, seed
    )


def _short_views(spec: TaskSpec, mod: Modality) -> Modality:
    if spec.modality != "text" or spec.train.max_len <= PRETRAIN_MAX_LEN:
        return mod
    short = spec.model_copy(update={"train": spec.train.model_copy(update={"max_len": PRETRAIN_MAX_LEN})})
    return get_modality(short, mod.data_dir)
