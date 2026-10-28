from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

from ..events import emit
from ..modalities import Modality
from ..model import SpecialistModel
from ..spec import TaskSpec
from ..spec.sections import SETFIT
from .common import head_of, sized_model, targets_of, to_device
from .contrastive import contrastive
from .evaluate import evaluate_rows
from .outputs import build_meta, finish, snapshot
from .rows import Prepared, prepare
from .trainer import class_weight
from .vectors import cached_embeddings

LOGISTIC_C = 1.0
LOGISTIC_MAX_ITER = 1000
ABSENT_LOGIT = -1e4
ENCODE_BATCH = 64


@torch.no_grad()
def pooled(model: SpecialistModel, mod: Modality, raws: list[str], dev: str) -> np.ndarray:
    model.eval()
    head, out = head_of(model), [np.zeros((0, model.backbone.dim), np.float32)]
    for i in range(0, len(raws), ENCODE_BATCH):
        inputs, _ = mod.collate(raws[i : i + ENCODE_BATCH])
        feats, mask = model.backbone(**to_device(inputs, dev))
        out.append(head.pooled(feats, mask).float().cpu().numpy())
    return np.concatenate(out)


def logistic_weights(clf: LogisticRegression, n_labels: int) -> tuple[np.ndarray, np.ndarray]:
    coef, bias = clf.coef_, clf.intercept_
    if len(clf.classes_) == 2:
        coef, bias = np.concatenate([-coef, coef]) / 2, np.concatenate([-bias, bias]) / 2
    weight = np.zeros((n_labels, coef.shape[1]), np.float32)
    shift = np.full(n_labels, ABSENT_LOGIT, np.float32)
    weight[clf.classes_], shift[clf.classes_] = coef, bias
    return weight, shift


@torch.no_grad()
def fit_head(spec: TaskSpec, model: SpecialistModel, emb: np.ndarray, y: list[int]) -> float:
    head = head_of(model)
    norm, linear = head.net[0], head.net[-1]
    norm.weight.fill_(norm.normalized_shape[0] ** -0.5)
    norm.bias.zero_()
    x = norm(torch.from_numpy(emb).to(norm.weight.device)).cpu().numpy()
    weight = class_weight(spec, y, "cpu").numpy()
    clf = LogisticRegression(C=LOGISTIC_C, max_iter=LOGISTIC_MAX_ITER, class_weight={c: weight[c] for c in set(y)})
    clf.fit(x, y)
    w, b = logistic_weights(clf, len(spec.labels))
    linear.weight.copy_(torch.from_numpy(w))
    linear.bias.copy_(torch.from_numpy(b))
    return float(log_loss(y, clf.predict_proba(x), labels=clf.classes_))


def _record(spec: TaskSpec, ev: dict[str, Any], loss: float, exp: str, log: Callable[[str], None]) -> dict:
    m = ev["metrics"]
    entry = {"epoch": 0, "loss": loss, "dev_objective": m["objective"], "dev_macro_f1": m["macro_f1"]}
    emit("epoch", **{**entry, "epoch": 1}, epochs=1, specialist=spec.name, experiment=exp)
    dev = f"dev {spec.objective.metric}={m['objective']:.4f} macro_f1={m['macro_f1']:.4f}"
    log(f"  logistic head: loss={loss:.4f} {dev}")
    return entry


def _train_embeddings(
    spec: TaskSpec, s: Prepared, model: SpecialistModel, raws: list[str], y: list[int], dev: str, log: Callable
) -> np.ndarray:
    if spec.train.method != SETFIT:
        return cached_embeddings(spec, raws, lambda xs: pooled(model, s.mod, xs, dev), log)
    contrastive(spec, s.mod, model, raws, y, s.rng, dev, log)
    return pooled(model, s.mod, raws, dev)


def fit_embedding(
    spec: TaskSpec,
    data_dir: Path,
    run_dir: Path,
    log: Callable[[str], None],
    seed: int | None,
    splits: tuple[str, ...] | list[str],
    dev: str,
) -> tuple[dict[str, Any], list[Any]]:
    s = prepare(spec, data_dir, log, seed, splits)
    log(f"  loading {len(s.train)} train / {len(s.dev)} dev examples")
    train_raw, dev_raw = s.mod.load_many(s.train), s.mod.load_many(s.dev)
    y = targets_of(spec, s.mod, s.train)
    model, n_params = sized_model(spec, s.mod, dev, log)
    model.to(dev)
    t0 = time.time()
    loss = fit_head(spec, model, _train_embeddings(spec, s, model, train_raw, y, dev, log), y)
    ev = evaluate_rows(model, s.mod, spec, s.dev, dev_raw, dev, None, calibrate=True)
    history = [_record(spec, ev, loss, run_dir.name, log)]
    best = snapshot(model, 0, ev)
    meta = build_meta(spec, best, history, [], n_params, dev, round(time.time() - t0, 1), s.counts())
    return finish(run_dir, s.ds, splits, meta, best)
