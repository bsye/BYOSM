"""Supervised training (after optional LeJEPA domain adaptation), dev-set model selection and calibration."""
from __future__ import annotations

import copy
import json
import math
import random
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

from . import jepa, metrics
from .config import device as pick_device
from .data.store import Dataset
from .modalities import IGNORE, Modality, get_modality
from .model import SpecialistModel, align_targets, count_params, task_loss
from .spec import TaskSpec


def targets_of(spec: TaskSpec, mod: Modality, rows: list[dict]) -> list[Any]:
    if spec.task_type == "frame":
        return [mod.frame_labels(r) for r in rows]
    idx = {n: i for i, n in enumerate(spec.label_names)}
    return [idx[r["label"]] for r in rows]


def batches(n: int, bs: int, shuffle: bool, rng: random.Random) -> list[list[int]]:
    order = list(range(n))
    if shuffle:
        rng.shuffle(order)
    return [order[i : i + bs] for i in range(0, n, bs)]


def to_device(inputs: dict[str, torch.Tensor], dev: str) -> dict[str, torch.Tensor]:
    return {k: v.to(dev) for k, v in inputs.items()}


@torch.no_grad()
def infer(model: SpecialistModel, mod: Modality, spec: TaskSpec, raws: list[Any], tgts: list[Any] | None, dev: str, head: str | None = None, bs: int = 64) -> dict[str, Any]:
    """Probabilities for every example (classify) or every valid frame (frame), plus per-example summaries."""
    model.to(dev).eval()
    probs, ys, per = [], [], []
    for b in batches(len(raws), bs, False, random.Random(0)):
        frame_lab = [tgts[i] for i in b] if (tgts is not None and spec.task_type == "frame") else None
        inputs, ft = mod.collate([raws[i] for i in b], frame_lab)
        logits, mask = model(to_device(inputs, dev), head)
        p = torch.softmax(logits.float(), -1).cpu()
        if spec.task_type == "frame":
            mask = mask.cpu()
            t = align_targets(ft, p.shape[1]) if ft is not None else None
            for j in range(len(b)):
                valid = mask[j] & (t[j] != IGNORE) if t is not None else mask[j]
                pj = p[j][valid].numpy()
                probs.append(pj)
                if t is not None:
                    yj = t[j][valid].numpy()
                    ys.append(yj)
                    per.append({"i": b[j], "error_rate": float((pj.argmax(-1) != yj).mean()) if len(yj) else 0.0})
        else:
            probs.append(p.numpy())
            if tgts is not None:
                ys.append(np.array([tgts[i] for i in b]))
    if spec.task_type == "frame":
        out = {"probs": np.concatenate(probs) if probs else np.zeros((0, len(spec.labels))), "per_example": per, "frame_probs": probs}
    else:
        out = {"probs": np.concatenate(probs), "per_example": []}
    if ys:
        out["y"] = np.concatenate(ys)
    return out


def evaluate_rows(model: SpecialistModel, mod: Modality, spec: TaskSpec, rows: list[dict], raws: list[Any], dev: str, threshold: float | None, calibrate: bool = False, head: str | None = None) -> dict[str, Any]:
    tg = targets_of(spec, mod, rows)
    out = infer(model, mod, spec, raws, tg, dev, head)
    thr = metrics.calibrate(spec, out["y"], out["probs"]) if calibrate else threshold
    m = metrics.compute(spec, out["y"], out["probs"], thr)
    errors = []
    if spec.task_type == "classify":
        pred = metrics.predict(spec, out["probs"], thr)
        for i, (r, yp, yt) in enumerate(zip(rows, pred, out["y"])):
            if yp != yt:
                errors.append({"id": r["id"], "input": r["input"], "gold": spec.label_names[yt], "pred": spec.label_names[yp], "confidence": float(out["probs"][i, yp])})
    else:
        for e in sorted(out["per_example"], key=lambda e: -e["error_rate"])[:200]:
            if e["error_rate"] > 0.05:
                r = rows[e["i"]]
                errors.append({"id": r["id"], "input": r["input"], "gold": "frames", "pred": f"{e['error_rate']:.0%} frames wrong", "confidence": e["error_rate"]})
    return {"metrics": m, "threshold": thr, "errors": errors}


def build_model(spec: TaskSpec, mod: Modality) -> SpecialistModel:
    return SpecialistModel(mod.build_backbone(), {spec.name: (len(spec.labels), spec.task_type)})


def fit(spec: TaskSpec, data_dir: Path, run_dir: Path, log: Callable[[str], None] = print, seed: int = 0) -> dict[str, Any]:
    torch.manual_seed(seed)
    rng = random.Random(seed)
    dev = pick_device()
    ds = Dataset(data_dir)
    mod = get_modality(spec, str(data_dir))
    train_rows, dev_rows = ds.load("train"), ds.load("dev")
    if not dev_rows:  # fall back to a train holdout
        rng.shuffle(train_rows)
        k = max(1, len(train_rows) // 10)
        dev_rows, train_rows = train_rows[:k], train_rows[k:]
    log(f"  loading {len(train_rows)} train / {len(dev_rows)} dev examples")
    train_raw = [mod.load(r) for r in train_rows]
    dev_raw = [mod.load(r) for r in dev_rows]
    train_t = targets_of(spec, mod, train_rows)

    model = build_model(spec, mod)
    n_params = count_params(model.backbone) + sum(count_params(h) for h in model.heads.values())
    log(f"  model: {spec.backbone} + {spec.task_type} head = {n_params / 1e6:.2f}M params on {dev}")
    tc = spec.train
    t0 = time.time()

    # Stage 1: LeJEPA on unlabeled in-domain inputs.
    jepa_hist: list[dict] = []
    if tc.pretrain_steps > 0:
        unl = [mod.load(r) for r in ds.unlabeled()] or train_raw
        log(f"  LeJEPA: {tc.pretrain_steps} steps on {len(unl)} unlabeled inputs (lambda={tc.pretrain_lambda})")
        jepa_hist = jepa.pretrain(model, mod, unl, tc.pretrain_steps, tc.batch_size, tc.lr_backbone, tc.pretrain_lambda, dev, log, seed)

    # Stage 2: supervised head (+ backbone fine-tune).
    model.to(dev)
    counts = np.bincount(np.concatenate([np.asarray(t).ravel() for t in train_t]) if spec.task_type == "frame" else np.array(train_t), minlength=len(spec.labels)).astype(float)
    weight = torch.tensor((counts.sum() / np.maximum(counts, 1) / len(counts)) ** 0.5, dtype=torch.float32, device=dev)
    for p in model.backbone.parameters():
        p.requires_grad = not tc.freeze_backbone
    groups = [{"params": [p for h in model.heads.values() for p in h.parameters()], "lr": tc.lr_head}]
    if not tc.freeze_backbone:
        groups.append({"params": list(model.backbone.parameters()), "lr": tc.lr_backbone})
    opt = torch.optim.AdamW(groups, weight_decay=0.01)
    steps = tc.epochs * math.ceil(len(train_raw) / tc.batch_size)
    warm = max(1, steps // 15)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(s, steps) / steps)))

    best: dict[str, Any] = {"objective": -1.0}
    history = []
    patience, bad = 4, 0
    for ep in range(tc.epochs):
        model.train()
        losses = []
        for b in batches(len(train_raw), tc.batch_size, True, rng):
            fl = [train_t[i] for i in b] if spec.task_type == "frame" else None
            inputs, ft = mod.collate([train_raw[i] for i in b], fl)
            logits, mask = model(to_device(inputs, dev))
            y = ft.to(dev) if ft is not None else torch.tensor([train_t[i] for i in b], device=dev)
            loss = task_loss(logits, y, mask, weight)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            losses.append(float(loss))
        ev = evaluate_rows(model, mod, spec, dev_rows, dev_raw, dev, None, calibrate=True)
        m = ev["metrics"]
        history.append({"epoch": ep, "loss": float(np.mean(losses)), "dev_objective": m["objective"], "dev_macro_f1": m["macro_f1"]})
        log(f"  epoch {ep + 1}/{tc.epochs} loss={np.mean(losses):.4f} dev {spec.objective.metric}={m['objective']:.4f} macro_f1={m['macro_f1']:.4f}")
        score = (m["objective"], m["macro_f1"])
        if score > (best["objective"], best.get("macro_f1", -1)):
            best = {"objective": m["objective"], "macro_f1": m["macro_f1"], "epoch": ep, "threshold": ev["threshold"], "state": copy.deepcopy({k: v.cpu() for k, v in model.state_dict().items()}), "dev": ev}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                log("  early stop")
                break

    run_dir.mkdir(parents=True, exist_ok=True)
    torch.save(best["state"], run_dir / "model.pt")
    spec.save(run_dir / "spec.yaml")
    meta = {
        "name": spec.name, "labels": spec.label_names, "threshold": best["threshold"], "best_epoch": best["epoch"],
        "dev": best["dev"]["metrics"], "params": n_params, "device": dev, "train_seconds": round(time.time() - t0, 1),
        "n_train": len(train_rows), "n_dev": len(dev_rows), "history": history, "jepa": jepa_hist[:: max(1, len(jepa_hist) // 50)],
    }
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    (run_dir / "dev_errors.json").write_text(json.dumps(best["dev"]["errors"], indent=2, ensure_ascii=False, default=str))
    return meta
