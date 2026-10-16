"""Metrics, objective, and decision-threshold calibration."""
from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support, roc_auc_score

from .spec import TaskSpec


def _binary(spec: TaskSpec) -> int | None:
    """Index of the positive label for thresholded binary decisions, else None."""
    if spec.positive_label and spec.positive_label in spec.label_names:
        return spec.label_names.index(spec.positive_label)
    if len(spec.labels) == 2 and spec.objective.metric in ("f1", "recall_at_precision", "precision_at_recall"):
        return 1
    return None


def predict(spec: TaskSpec, probs: np.ndarray, threshold: float | None) -> np.ndarray:
    pos = _binary(spec)
    if pos is None or threshold is None:
        return probs.argmax(-1)
    neg = probs.copy()
    neg[:, pos] = -1
    return np.where(probs[:, pos] >= threshold, pos, neg.argmax(-1))


def _prf_at(y: np.ndarray, score: np.ndarray, thr: float) -> tuple[float, float, float]:
    pred = score >= thr
    tp = float((pred & y).sum())
    p = tp / max(1.0, float(pred.sum()))
    r = tp / max(1.0, float(y.sum()))
    return p, r, 2 * p * r / max(1e-9, p + r)


def calibrate(spec: TaskSpec, y: np.ndarray, probs: np.ndarray) -> float | None:
    """Pick the decision threshold on the dev set that optimizes the spec's objective."""
    pos = _binary(spec)
    if pos is None:
        return None
    yb, score = y == pos, probs[:, pos]
    cands = np.unique(np.concatenate([np.quantile(score, np.linspace(0, 1, 401)), [0.5]]))
    best, best_thr = -1.0, 0.5
    m, c = spec.objective.metric, spec.objective.constraint
    for thr in cands:
        p, r, f = _prf_at(yb, score, thr)
        if m == "recall_at_precision":
            v = r if p >= c else -1 + p  # infeasible thresholds rank by how close they get
        elif m == "precision_at_recall":
            v = p if r >= c else -1 + r
        else:
            v = f
        if v > best:
            best, best_thr = v, float(thr)
    return best_thr


def compute(spec: TaskSpec, y: np.ndarray, probs: np.ndarray, threshold: float | None) -> dict[str, Any]:
    names = spec.label_names
    pred = predict(spec, probs, threshold)
    labels = list(range(len(names)))
    p, r, f, s = precision_recall_fscore_support(y, pred, labels=labels, zero_division=0)
    out: dict[str, Any] = {
        "n": int(len(y)),
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, labels=labels, average="macro", zero_division=0)),
        "per_label": {names[i]: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]), "support": int(s[i])} for i in labels},
        "confusion": confusion_matrix(y, pred, labels=labels).tolist(),
        "threshold": threshold,
    }
    pos = _binary(spec)
    if pos is not None:
        yb = y == pos
        out.update(precision=float(p[pos]), recall=float(r[pos]), f1=float(f[pos]))
        if 0 < yb.sum() < len(yb):
            out["roc_auc"] = float(roc_auc_score(yb, probs[:, pos]))
    out["objective"] = objective_value(spec, out)
    return out


def objective_value(spec: TaskSpec, m: dict[str, Any]) -> float:
    o = spec.objective
    if o.metric == "recall_at_precision":
        return m.get("recall", 0.0) if m.get("precision", 0.0) >= o.constraint else 0.0
    if o.metric == "precision_at_recall":
        return m.get("precision", 0.0) if m.get("recall", 0.0) >= o.constraint else 0.0
    if o.metric == "f1":
        return m.get("f1", m["macro_f1"])
    return float(m[o.metric])
