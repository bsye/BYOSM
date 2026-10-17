"""Supervised training (after optional LeJEPA domain adaptation), dev-set model selection and calibration."""
from __future__ import annotations

import contextlib
import copy
import json
import math
import random
import re
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

from . import jepa, metrics
from .config import device as pick_device
from .data.store import Dataset
from .modalities import IGNORE, Modality, get_modality
from .model import SpecialistModel, align_probs, align_targets, count_params, distill_loss, task_loss
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


_LAYER = re.compile(r"\.(?:layer|layers|blocks|block|h|stages)\.(\d+)\.")


def pretrained_encoders(model: torch.nn.Module) -> dict[str, torch.nn.Module]:
    """Top-level Hugging Face encoders inside the model (the parts that start from pretrained weights)."""
    from transformers import PreTrainedModel

    out: dict[str, torch.nn.Module] = {}
    for name, m in model.named_modules():
        if isinstance(m, PreTrainedModel) and not any(name.startswith(p + ".") for p in out):
            out[name] = m
    return out


def param_groups(model: SpecialistModel, tc: Any) -> list[dict]:
    """Pretrained encoders: layer-wise LR decay, lr_backbone at the top layer down to lr_backbone * layer_decay_floor at the
    embeddings (low layers hold general features worth keeping). Layers trained from scratch (heads, projections, fusion)
    get lr_head. A backbone with no pretrained encoder (tiny-crnn, tiny-cnn) trains at lr_backbone."""
    enc = pretrained_encoders(model.backbone)
    groups: dict[float, list] = {}
    seen: set[int] = set()
    for m in enc.values():
        named = list(m.named_parameters())
        depth = {n: (int(g.group(1)) + 1 if (g := _LAYER.search("." + n)) else (0 if "embed" in n else None)) for n, _ in named}
        top = max([d for d in depth.values() if d is not None], default=0) + 1
        decay = tc.layer_decay_floor ** (1 / top) if top > 0 else 1.0
        for n, p in named:
            d = top if depth[n] is None else depth[n]
            groups.setdefault(tc.lr_backbone * decay ** (top - d), []).append(p)
            seen.add(id(p))
    scratch = tc.lr_head if enc else tc.lr_backbone
    for p in model.backbone.parameters():
        if id(p) not in seen:
            groups.setdefault(scratch, []).append(p)
    for h in model.heads.values():
        groups.setdefault(tc.lr_head, []).extend(h.parameters())
    return [{"params": ps, "lr": lr} for lr, ps in groups.items()]


def amp_dtype(spec: TaskSpec, model: torch.nn.Module, dev: str) -> torch.dtype | None:
    """Mixed precision: measured 2.1x faster per step (fp16, MPS) on DiT+MiniLM. 'auto' uses it only for pretrained encoders;
    small from-scratch recurrent nets keep fp32."""
    p = spec.train.precision
    if p == "auto":
        if dev not in ("mps", "cuda") or not pretrained_encoders(model):
            return None
        return torch.bfloat16 if dev == "cuda" and torch.cuda.is_bf16_supported() else torch.float16
    return {"fp32": None, "bf16": torch.bfloat16, "fp16": torch.float16}[p]


def autocast(dev: str, dtype: torch.dtype | None) -> contextlib.AbstractContextManager:
    return torch.autocast(dev, dtype=dtype) if dtype is not None else contextlib.nullcontext()


@torch.no_grad()
def infer(model: SpecialistModel, mod: Modality, spec: TaskSpec, raws: list[Any], tgts: list[Any] | None, dev: str, head: str | None = None, bs: int = 64) -> dict[str, Any]:
    """Probabilities for every example (classify) or every valid frame (frame), plus per-example summaries."""
    model.to(dev).eval()
    amp = amp_dtype(spec, model, dev)
    probs, ys, per = [], [], []
    for b in batches(len(raws), bs, False, random.Random(0)):
        frame_lab = [tgts[i] for i in b] if (tgts is not None and spec.task_type == "frame") else None
        inputs, ft = mod.collate([raws[i] for i in b], frame_lab)
        with autocast(dev, amp):
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
    return SpecialistModel(mod.build_backbone(), {spec.name: (len(spec.labels), spec.task_type, spec.train.pool)})


def _teacher_targets(spec: TaskSpec, mod: Modality, teacher_dir: Path, train_rows: list[dict], ds: Dataset, data_dir: Path, dev: str,
                     log: Callable[[str], None]) -> tuple[list[Any], list[Any], list[Any]]:
    """Teacher probabilities for every training input, plus extra unlabeled inputs (targets: IGNORE + teacher only)."""
    from .predictors import ExperimentPredictor

    tspec = TaskSpec.load(teacher_dir / "spec.yaml")
    teacher = ExperimentPredictor(tspec, teacher_dir, teacher_dir.name, device=dev if spec.task_type == "classify" else "cpu")
    tmod = get_modality(tspec, str(data_dir))
    seen = {str(r["input"]) for r in train_rows}
    extra = [r for r in ds.unlabeled() if r.get("source") != "train-inputs" and str(r["input"]) not in seen][: spec.train.distill_unlabeled]
    rows = train_rows + extra
    log(f"  distill: teacher {teacher_dir.name} ({teacher.params / 1e6:.1f}M) labels {len(train_rows)} train + {len(extra)} unlabeled inputs")
    if spec.task_type == "frame":
        soft = [teacher.frame_probs(tmod.load(r)) for r in rows]
        extra_t = [[IGNORE] * len(s) for s in soft[len(train_rows):]]
    else:
        soft = list(teacher.probs_batch(tmod.load_many(rows)))
        extra_t = [IGNORE] * len(extra)
    return soft, mod.load_many(extra), extra_t


def fit(spec: TaskSpec, data_dir: Path, run_dir: Path, log: Callable[[str], None] = print, seed: int | None = None) -> dict[str, Any]:
    seed = spec.train.seed if seed is None else seed
    torch.manual_seed(seed)
    rng = random.Random(seed)
    dev = pick_device()
    ds = Dataset(data_dir)
    mod = get_modality(spec, str(data_dir))
    train_rows, dev_rows = ds.load("train"), ds.load("dev")
    n_folded = 0
    if spec.trains_on_test_split():
        extra = ds.load("test")
        n_folded = len(extra)
        log(f"  primary suite is external: folding the builtin test split into training (+{len(extra)})")
        train_rows += extra
    if not dev_rows:  # fall back to a train holdout
        rng.shuffle(train_rows)
        k = max(1, len(train_rows) // 10)
        dev_rows, train_rows = train_rows[:k], train_rows[k:]
    log(f"  loading {len(train_rows)} train / {len(dev_rows)} dev examples")
    train_raw = mod.load_many(train_rows)
    dev_raw = mod.load_many(dev_rows)
    train_t = targets_of(spec, mod, train_rows)

    model = build_model(spec, mod)
    n_params = count_params(model.backbone) + sum(count_params(h) for h in model.heads.values())
    log(f"  model: {spec.backbone} + {spec.task_type} head = {n_params / 1e6:.2f}M params on {dev}")
    tc = spec.train
    t0 = time.time()

    # Stage 1: LeJEPA on unlabeled in-domain inputs.
    jepa_hist: list[dict] = []
    if tc.pretrain_steps > 0:
        unl = mod.load_many(ds.unlabeled()) or train_raw
        log(f"  LeJEPA: {tc.pretrain_steps} steps on {len(unl)} unlabeled inputs (lambda={tc.pretrain_lambda})")
        jepa_hist = jepa.pretrain(model, mod, unl, tc.pretrain_steps, tc.batch_size, tc.lr_backbone, tc.pretrain_lambda, dev, log, seed)

    # Distillation: teacher soft targets on the training inputs + extra unlabeled inputs (teacher-only targets).
    soft: list[Any] | None = None
    n_labeled = len(train_raw)
    if tc.distill_from:
        soft, extra_raw, extra_t = _teacher_targets(spec, mod, run_dir.parent / tc.distill_from, train_rows, ds, data_dir, dev, log)
        train_raw += extra_raw
        train_t += extra_t

    # Stage 2: supervised head (+ backbone fine-tune).
    model.to(dev)
    gold = train_t[:n_labeled]
    counts = np.bincount(np.concatenate([np.asarray(t).ravel() for t in gold]) if spec.task_type == "frame" else np.array(gold), minlength=len(spec.labels)).astype(float)
    weight = torch.tensor((counts.sum() / np.maximum(counts, 1) / len(counts)) ** 0.5, dtype=torch.float32, device=dev)
    enc = pretrained_encoders(model.backbone)
    frozen = list(enc.values()) if enc else [model.backbone]  # freeze_backbone / probe epochs freeze these

    def freeze(on: bool) -> None:
        for m in frozen:
            for p in m.parameters():
                p.requires_grad = not on

    opt = torch.optim.AdamW(param_groups(model, tc), weight_decay=0.01)
    steps_per_epoch = math.ceil(len(train_raw) / tc.batch_size)
    steps = tc.epochs * steps_per_epoch
    warm = max(1, steps // 15)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(s, steps) / steps)))
    amp = amp_dtype(spec, model, dev)
    scaler = torch.amp.GradScaler(dev, enabled=amp == torch.float16)
    probe = 0 if tc.freeze_backbone or not enc else min(tc.probe_epochs, tc.epochs // 4)  # at most a quarter of the run
    # EMA over ~half an epoch of steps: smooths SGD noise without lagging behind the cosine schedule.
    ema = torch.optim.swa_utils.AveragedModel(model, multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(1 - 1 / max(10, steps_per_epoch / 2)), use_buffers=True) if tc.ema else None
    if probe or amp or ema:
        log(f"  recipe: {probe} probe epoch(s), layer LR floor {tc.layer_decay_floor}, ema={'on' if ema else 'off'}, precision={str(amp).replace('torch.', '') if amp else 'fp32'}")

    best: dict[str, Any] = {"objective": -1.0}
    history = []
    patience, bad = 4, 0
    for ep in range(tc.epochs):
        frz = tc.freeze_backbone or ep < probe
        freeze(frz)
        model.train()
        for m in frozen if frz else []:
            m.eval()  # a frozen encoder is a fixed feature extractor: no dropout, no BatchNorm statistics updates
        losses = []
        for b in batches(len(train_raw), tc.batch_size, True, rng):
            fl = [train_t[i] for i in b] if spec.task_type == "frame" else None
            inputs, ft = mod.collate([train_raw[i] for i in b], fl)
            with autocast(dev, amp):
                logits, mask = model(to_device(inputs, dev))
            logits = logits.float()
            y = ft.to(dev) if ft is not None else torch.tensor([train_t[i] for i in b], device=dev)
            has_gold = bool((y != IGNORE).any())
            loss = task_loss(logits, y, mask, weight) * (tc.distill_alpha if soft is not None else 1.0) if has_gold else logits.sum() * 0
            if soft is not None:
                if spec.task_type == "frame":
                    tp = torch.nn.utils.rnn.pad_sequence([torch.as_tensor(soft[i]) for i in b], batch_first=True).to(dev)
                    lens = torch.tensor([len(soft[i]) for i in b], device=dev)
                    valid = (torch.arange(tp.shape[1], device=dev)[None] < lens[:, None]).float()
                    tp, valid = align_probs(tp, logits.shape[1]), align_probs(valid[..., None], logits.shape[1])[..., 0] * mask.float()
                else:
                    tp, valid = torch.as_tensor(np.stack([soft[i] for i in b])).to(dev), torch.ones(len(b), device=dev)
                loss = loss + (1 - tc.distill_alpha) * distill_loss(logits.float(), tp.float(), valid, tc.distill_temperature)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            if ema is not None:
                ema.update_parameters(model)
            losses.append(float(loss))
        sel = ema.module if ema is not None else model
        ev = evaluate_rows(sel, mod, spec, dev_rows, dev_raw, dev, None, calibrate=True)
        m = ev["metrics"]
        history.append({"epoch": ep, "loss": float(np.mean(losses)), "dev_objective": m["objective"], "dev_macro_f1": m["macro_f1"]})
        log(f"  epoch {ep + 1}/{tc.epochs} loss={np.mean(losses):.4f} dev {spec.objective.metric}={m['objective']:.4f} macro_f1={m['macro_f1']:.4f}")
        score = (m["objective"], m["macro_f1"])
        if score > (best["objective"], best.get("macro_f1", -1)):
            best = {"objective": m["objective"], "macro_f1": m["macro_f1"], "epoch": ep, "threshold": ev["threshold"], "state": copy.deepcopy({k: v.cpu() for k, v in sel.state_dict().items()}), "dev": ev}
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
        "distilled_from": tc.distill_from, "n_teacher_only": len(train_raw) - n_labeled, "n_test_folded": n_folded,
    }
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    (run_dir / "dev_errors.json").write_text(json.dumps(best["dev"]["errors"], indent=2, ensure_ascii=False, default=str))
    return meta
