from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import torch

from ..data.store import BASE_TRAIN, Dataset
from ..modalities import Modality, get_modality
from ..spec import TaskSpec


def split_rows(
    spec: TaskSpec,
    ds: Dataset,
    rng: random.Random,
    log: Callable[[str], None],
    splits: tuple[str, ...] | list[str] = BASE_TRAIN,
) -> tuple[list[dict], list[dict], int]:
    train_rows, dev_rows = [r for s in splits for r in ds.load(s)], ds.load("dev")
    n_folded = 0
    if spec.trains_on_test_split():
        extra = ds.load("test")
        n_folded = len(extra)
        log(f"  primary suite is external: folding the builtin test split into training (+{len(extra)})")
        train_rows += extra
    if not dev_rows:
        rng.shuffle(train_rows)
        k = max(1, len(train_rows) // 10)
        dev_rows, train_rows = train_rows[:k], train_rows[k:]
    return train_rows, dev_rows, n_folded


@dataclass
class Prepared:
    ds: Dataset
    mod: Modality
    rng: random.Random
    seed: int
    train: list[dict]
    dev: list[dict]
    n_folded: int

    def counts(self, n_teacher_only: int = 0) -> dict[str, int]:
        return {
            "n_train": len(self.train),
            "n_dev": len(self.dev),
            "n_teacher_only": n_teacher_only,
            "n_test_folded": self.n_folded,
        }


def prepare(
    spec: TaskSpec, data_dir: Path, log: Callable[[str], None], seed: int | None, splits: tuple[str, ...] | list[str]
) -> Prepared:
    seed = spec.train.seed if seed is None else seed
    torch.manual_seed(seed)
    rng = random.Random(seed)
    ds = Dataset(data_dir)
    mod = get_modality(spec, str(data_dir))
    train_rows, dev_rows, n_folded = split_rows(spec, ds, rng, log, splits)
    return Prepared(ds, mod, rng, seed, train_rows, dev_rows, n_folded)
