from __future__ import annotations

import random
from collections.abc import Callable
from typing import Any

import torch

from ..project import WEIGHTS
from ..spec import TaskSpec

REPLAY_SHARE = 0.5
CONTINUE_LR_FACTOR = 0.5
CONTINUE_EPOCHS = 3


def _parent(spec: TaskSpec) -> tuple[Any, TaskSpec]:
    from ..project import Project

    p = Project(spec.name)
    return p, p.exp_spec(spec.train.init_from)


def _rows_by_label(
    old: torch.Tensor, new: torch.Tensor, old_labels: list[str], labels: list[str]
) -> torch.Tensor | None:
    if old.shape[1:] != new.shape[1:] or old.shape[0] != len(old_labels) or new.shape[0] != len(labels):
        return None
    out = new.clone()
    for i, lab in enumerate(labels):
        if lab in old_labels:
            out[i] = old[old_labels.index(lab)]
    return out


def warm_start(model: torch.nn.Module, spec: TaskSpec, log: Callable[[str], None]) -> None:
    p, parent = _parent(spec)
    state = torch.load(p.exp_dir(spec.train.init_from) / WEIGHTS, map_location="cpu")
    own = model.state_dict()
    kept, mapped = 0, 0
    for key, value in state.items():
        if key not in own:
            continue
        if own[key].shape == value.shape:
            own[key], kept = value, kept + 1
        elif (rows := _rows_by_label(value, own[key], parent.label_names, spec.label_names)) is not None:
            own[key], mapped = rows, mapped + 1
    model.load_state_dict(own)
    new = [lab for lab in spec.label_names if lab not in parent.label_names]
    extra = f"; new labels start fresh: {', '.join(new)}" if new else ""
    log(f"  continuing from {spec.train.init_from}: {kept} weights kept, {mapped} label rows mapped{extra}")


def _known(label: Any, spec: TaskSpec) -> bool:
    return not isinstance(label, str) or label in spec.label_names


def replay(
    spec: TaskSpec, train_rows: list[dict], held: list[dict], rng: random.Random, log: Callable[[str], None]
) -> list[dict]:
    from ..data.store import Dataset

    p, parent = _parent(spec)
    if parent.data_key() == spec.data_key():
        return []
    seen = {r["input"] for r in held + train_rows}
    old = [
        r
        for r in Dataset(p.data_dir_for(parent)).load("train")
        if r["input"] not in seen and r["label"] in spec.label_names
    ]
    rng.shuffle(old)
    picked = old[: int(len(train_rows) * REPLAY_SHARE)]
    log(f"  replaying {len(picked)} rows {spec.train.init_from} trained on, so it keeps what it learned")
    return picked


def continued(parent: TaskSpec, parent_exp: str, epochs: int | None = None) -> TaskSpec:
    tc = parent.train
    changes = {
        "init_from": parent_exp,
        "lr_backbone": tc.lr_backbone * CONTINUE_LR_FACTOR,
        "epochs": epochs or min(tc.epochs, CONTINUE_EPOCHS),
    }
    return parent.model_copy(update={"train": tc.model_copy(update=changes)})
