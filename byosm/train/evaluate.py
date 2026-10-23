from __future__ import annotations

import random
from typing import Any

import numpy as np
import torch

from .. import metrics
from ..modalities import IGNORE, Modality
from ..model import SpecialistModel, align_targets
from ..spec import TaskSpec
from .common import amp_dtype, autocast, batches, targets_of, to_device


def _frame_batch(p: torch.Tensor, mask: torch.Tensor, ft: Any, b: list[int], probs: list, ys: list, per: list) -> None:
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


@torch.no_grad()
def infer(
    model: SpecialistModel,
    mod: Modality,
    spec: TaskSpec,
    raws: list[Any],
    tgts: list[Any] | None,
    dev: str,
    head: str | None = None,
    bs: int = 64,
) -> dict[str, Any]:
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
            _frame_batch(p, mask, ft, b, probs, ys, per)
        else:
            probs.append(p.numpy())
            if tgts is not None:
                ys.append(np.array([tgts[i] for i in b]))
    if spec.task_type == "frame":
        stacked = np.concatenate(probs) if probs else np.zeros((0, len(spec.labels)))
        out = {"probs": stacked, "per_example": per, "frame_probs": probs}
    else:
        out = {"probs": np.concatenate(probs), "per_example": []}
    if ys:
        out["y"] = np.concatenate(ys)
    return out


def _classify_errors(spec: TaskSpec, rows: list[dict], out: dict[str, Any], thr: Any) -> list[dict]:
    errors = []
    pred = metrics.predict(spec, out["probs"], thr)
    for i, (r, yp, yt) in enumerate(zip(rows, pred, out["y"], strict=False)):
        if yp != yt:
            errors.append(
                {
                    "id": r["id"],
                    "input": r["input"],
                    "gold": spec.label_names[yt],
                    "pred": spec.label_names[yp],
                    "confidence": float(out["probs"][i, yp]),
                }
            )
    return errors


def _frame_errors(rows: list[dict], out: dict[str, Any]) -> list[dict]:
    errors = []
    for e in sorted(out["per_example"], key=lambda e: -e["error_rate"])[:200]:
        if e["error_rate"] > 0.05:
            r = rows[e["i"]]
            pred = f"{e['error_rate']:.0%} frames wrong"
            errors.append(
                {"id": r["id"], "input": r["input"], "gold": "frames", "pred": pred, "confidence": e["error_rate"]}
            )
    return errors


def evaluate_rows(
    model: SpecialistModel,
    mod: Modality,
    spec: TaskSpec,
    rows: list[dict],
    raws: list[Any],
    dev: str,
    threshold: float | None,
    calibrate: bool = False,
    head: str | None = None,
) -> dict[str, Any]:
    tg = targets_of(spec, mod, rows)
    out = infer(model, mod, spec, raws, tg, dev, head)
    if not np.isfinite(np.asarray(out["probs"], dtype=float)).all():
        raise RuntimeError(
            f"training diverged: the model outputs NaN on {dev}; try train.precision=fp32 or a lower learning rate"
        )
    thr = metrics.calibrate(spec, out["y"], out["probs"]) if calibrate else threshold
    lengths = [len(p) for p in out["frame_probs"]] if "frame_probs" in out else None
    m = metrics.compute(spec, out["y"], out["probs"], thr, lengths)
    errors = _classify_errors(spec, rows, out, thr) if spec.task_type == "classify" else _frame_errors(rows, out)
    return {"metrics": m, "threshold": thr, "errors": errors}
