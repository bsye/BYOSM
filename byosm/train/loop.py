from __future__ import annotations

import json
import random
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .. import jepa
from ..config import device as pick_device
from ..data.store import BASE_TRAIN, Dataset
from ..events import emit, stage
from ..modalities import Modality, get_modality
from ..model import SpecialistModel, count_params
from ..spec import TaskSpec
from .common import build_model, cpu_state, targets_of
from .evaluate import evaluate_rows
from .gpu import gpu_slot, shrink_on_oom
from .outputs import build_meta, save_run
from .teacher import _teacher_targets, row_soft_targets
from .trainer import Trainer, class_weight

PATIENCE = 4


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


def _pretrain(
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
    return jepa.pretrain(
        model, mod, unl, tc.pretrain_steps, tc.batch_size, tc.lr_backbone, tc.pretrain_lambda, dev, log, seed
    )


def _dev_epoch(
    trainer: Trainer,
    ep: int,
    losses: list[float],
    dev_rows: list[dict],
    dev_raw: list[Any],
    history: list[dict],
    log: Callable[[str], None],
    exp: str,
) -> dict[str, Any]:
    spec = trainer.spec
    ev = evaluate_rows(trainer.selected, trainer.mod, spec, dev_rows, dev_raw, trainer.dev, None, calibrate=True)
    m = ev["metrics"]
    loss = float(np.mean(losses))
    history.append({"epoch": ep, "loss": loss, "dev_objective": m["objective"], "dev_macro_f1": m["macro_f1"]})
    emit("epoch", **{**history[-1], "epoch": ep + 1}, epochs=spec.train.epochs, specialist=spec.name, experiment=exp)
    log(
        f"  epoch {ep + 1}/{spec.train.epochs} loss={np.mean(losses):.4f} "
        f"dev {spec.objective.metric}={m['objective']:.4f} macro_f1={m['macro_f1']:.4f}"
    )
    return ev


def _snapshot(sel: torch.nn.Module, ep: int, ev: dict[str, Any]) -> dict[str, Any]:
    m = ev["metrics"]
    state = cpu_state(sel)
    return {
        "objective": m["objective"],
        "macro_f1": m["macro_f1"],
        "epoch": ep,
        "threshold": ev["threshold"],
        "state": state,
        "dev": ev,
    }


def _select(
    trainer: Trainer, dev_rows: list[dict], dev_raw: list[Any], log: Callable[[str], None], exp: str
) -> tuple[dict[str, Any], list[dict]]:
    tc = trainer.spec.train
    best: dict[str, Any] = {"objective": -1.0}
    history = []
    bad = 0
    for ep in range(tc.epochs):
        losses = trainer.epoch(ep)
        ev = _dev_epoch(trainer, ep, losses, dev_rows, dev_raw, history, log, exp)
        m = ev["metrics"]
        if (m["objective"], m["macro_f1"]) > (best["objective"], best.get("macro_f1", -1)):
            best = _snapshot(trainer.selected, ep, ev)
            bad = 0
        else:
            bad += 1
            if bad >= PATIENCE:
                log("  early stop")
                break
    return best, history


def _build(spec: TaskSpec, mod: Modality, dev: str, log: Callable[[str], None]) -> tuple[SpecialistModel, int]:
    model = build_model(spec, mod)
    n_params = count_params(model.backbone) + sum(count_params(h) for h in model.heads.values())
    log(f"  model: {spec.backbone} + {spec.task_type} head = {n_params / 1e6:.2f}M params on {dev}")
    return model, n_params


@stage("Train")
def fit(
    spec: TaskSpec,
    data_dir: Path,
    run_dir: Path,
    log: Callable[[str], None] = print,
    seed: int | None = None,
    splits: tuple[str, ...] | list[str] = BASE_TRAIN,
) -> tuple[dict[str, Any], list[Any]]:
    dev = pick_device()
    with gpu_slot(dev, log):
        return shrink_on_oom(spec, lambda s: _fit(s, data_dir, run_dir, log, seed, splits, dev), log, dev)


def _fit(
    spec: TaskSpec,
    data_dir: Path,
    run_dir: Path,
    log: Callable[[str], None],
    seed: int | None,
    splits: tuple[str, ...] | list[str],
    dev: str,
) -> tuple[dict[str, Any], list[Any]]:
    seed = spec.train.seed if seed is None else seed
    torch.manual_seed(seed)
    rng = random.Random(seed)
    ds = Dataset(data_dir)
    mod = get_modality(spec, str(data_dir))
    train_rows, dev_rows, n_folded = split_rows(spec, ds, rng, log, splits)
    log(f"  loading {len(train_rows)} train / {len(dev_rows)} dev examples")
    train_raw = mod.load_many(train_rows)
    dev_raw = mod.load_many(dev_rows)
    train_t = targets_of(spec, mod, train_rows)
    model, n_params = _build(spec, mod, dev, log)
    tc = spec.train
    t0 = time.time()
    jepa_hist = _pretrain(spec, model, mod, ds, train_raw, dev, log, seed)
    soft: list[Any] | None = row_soft_targets(spec, train_rows)
    n_labeled = len(train_raw)
    if tc.distill_from:
        teacher_dir = run_dir.parent / tc.distill_from
        soft, extra_raw, extra_t = _teacher_targets(spec, mod, teacher_dir, train_rows, ds, data_dir, dev, log)
        train_raw += extra_raw
        train_t += extra_t
    model.to(dev)
    weight = class_weight(spec, train_t[:n_labeled], dev)
    trainer = Trainer(spec, mod, model, dev, train_raw, train_t, soft, weight, rng)
    trainer.log_recipe(log)
    best, history = _select(trainer, dev_rows, dev_raw, log, run_dir.name)
    counts = {
        "n_train": len(train_rows),
        "n_dev": len(dev_rows),
        "n_teacher_only": len(train_raw) - n_labeled,
        "n_test_folded": n_folded,
    }
    meta = build_meta(spec, best, history, jepa_hist, n_params, dev, round(time.time() - t0, 1), counts)
    meta["train_splits"] = list(splits)
    save_run(run_dir, best)
    return meta, json.loads(json.dumps(best["dev"]["errors"], default=str))
