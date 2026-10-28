from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from ..config import device as pick_device
from ..data.store import BASE_TRAIN, CALIB
from ..events import emit, stage
from ..spec import TaskSpec
from ..spec.sections import FINETUNE
from .common import sized_model
from .embedding import fit_embedding
from .evaluate import evaluate_rows
from .gpu import gpu_slot
from .gpu_memory import release_cache
from .outputs import build_meta, finish, snapshot
from .pretraining import pretrain_backbone
from .recovery import DivergedError, recover
from .regularise import reinit_top
from .rows import prepare
from .teacher import _teacher_targets, pseudo_labels, row_soft_targets
from .trainer import Trainer, class_weight
from .warm import replay, warm_start

PATIENCE = 4
COLLAPSE_FROM = 0.4
COLLAPSE_SHARE = 0.5


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
        release_cache()
        m = ev["metrics"]
        if best.get("macro_f1", 0) >= COLLAPSE_FROM and m["macro_f1"] < best["macro_f1"] * COLLAPSE_SHARE:
            raise DivergedError(
                f"training collapsed: dev macro F1 fell from {best['macro_f1']:.3f} to {m['macro_f1']:.3f}"
            )
        if (m["objective"], m["macro_f1"]) > (best["objective"], best.get("macro_f1", -1)):
            best = snapshot(trainer.selected, ep, ev)
            bad = 0
        else:
            bad += 1
            if bad >= PATIENCE:
                log("  early stop")
                break
    return best, history


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
    train = _fit if spec.train.method == FINETUNE else fit_embedding
    with gpu_slot(dev, log, key=spec.backbone or spec.name):
        return recover(spec, lambda s: train(s, data_dir, run_dir, log, seed, splits, dev), log, dev)


def lab_hooks(spec: TaskSpec, log: Callable[[str], None]) -> dict[str, Callable[..., Any]]:
    if not spec.train.lab_method:
        return {}
    from ..lab.plugin import load
    from ..project import Project

    hooks = load(Project(spec.name), spec.train.lab_method)
    log(f"  lab method {spec.train.lab_method}: {', '.join(hooks) or 'no hooks'}")
    return hooks


def _fit(
    spec: TaskSpec,
    data_dir: Path,
    run_dir: Path,
    log: Callable[[str], None],
    seed: int | None,
    splits: tuple[str, ...] | list[str],
    dev: str,
) -> tuple[dict[str, Any], list[Any]]:
    s = prepare(spec, data_dir, log, seed, splits)
    hooks = lab_hooks(spec, log)
    if "rows" in hooks:
        s.train = hooks["rows"](s.train, s.rng)
    if spec.train.init_from:
        s.train += replay(spec, s.train, s.dev + s.ds.load(CALIB) + s.ds.load("test"), s.rng, log)
    if spec.train.self_train > 0:
        s.train += pseudo_labels(spec, s.train, s.ds, data_dir, dev, log)
    train_raw, dev_raw, train_t = s.loaded(spec, log)
    model, n_params = sized_model(spec, s.mod, dev, log)
    if spec.train.init_from:
        warm_start(model, spec, log)
    if spec.train.reinit_layers:
        reinit_top(model, spec.train.reinit_layers, log)
    tc = spec.train
    t0 = time.time()
    jepa_hist = pretrain_backbone(spec, model, s.mod, s.ds, train_raw, dev, log, s.seed)
    soft: list[Any] | None = row_soft_targets(spec, s.train)
    n_labeled = len(train_raw)
    if tc.distill_from:
        teacher_dir = run_dir.parent / tc.distill_from
        soft, extra_raw, extra_t = _teacher_targets(spec, s.mod, teacher_dir, s.train, s.ds, data_dir, dev, log)
        train_raw += extra_raw
        train_t += extra_t
    model.to(dev)
    weight = class_weight(spec, train_t[:n_labeled], dev)
    trainer = Trainer(spec, s.mod, model, dev, train_raw, train_t, soft, weight, s.rng, hooks)
    trainer.log_recipe(log)
    best, history = _select(trainer, s.dev, dev_raw, log, run_dir.name)
    seconds = round(time.time() - t0, 1)
    meta = build_meta(spec, best, history, jepa_hist, n_params, dev, seconds, s.counts(len(train_raw) - n_labeled))
    return finish(run_dir, s.ds, splits, meta, best)
