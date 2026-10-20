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


# ------------------------------------------------------------------ units + paired bootstrap
# A scored split is kept as units: one (gold, decision) pair per item for classify, one confusion matrix per file for frame
# tasks (frames of one file are not independent). Every objective is a function of the summed confusion matrix, so
# resampling units gives confidence intervals and paired "is A really better than B" tests.


def units(n_labels: int, y: np.ndarray, pred: np.ndarray, groups: list[int] | None = None) -> dict[str, Any]:
    if groups is None:
        return {"y": np.asarray(y).astype(int).tolist(), "pred": np.asarray(pred).astype(int).tolist()}
    cms, s = [], 0
    for n in groups:
        cms.append(confusion_matrix(y[s : s + n], pred[s : s + n], labels=list(range(n_labels))).ravel().tolist())
        s += n
    return {"cm": cms}


def _cms(u: dict[str, Any], n_labels: int) -> np.ndarray:
    """-> [n_units, C, C]"""
    if "cm" in u:
        return np.asarray(u["cm"], dtype=np.int64).reshape(-1, n_labels, n_labels)
    out = np.zeros((len(u["y"]), n_labels, n_labels), np.int64)
    out[np.arange(len(u["y"])), u["y"], u["pred"]] = 1
    return out


def objective_from_cm(spec: TaskSpec, cm: np.ndarray) -> float:
    tp = np.diag(cm).astype(float)
    p = tp / np.maximum(cm.sum(0), 1)
    r = tp / np.maximum(cm.sum(1), 1)
    f = 2 * p * r / np.maximum(p + r, 1e-9)
    m: dict[str, Any] = {"accuracy": tp.sum() / max(1, cm.sum()), "macro_f1": float(f.mean())}
    pos = _binary(spec)
    if pos is not None:
        m.update(precision=p[pos], recall=r[pos], f1=f[pos])
    return objective_value(spec, m)


def bootstrap(spec: TaskSpec, a: dict[str, Any], b: dict[str, Any] | None = None, n: int = 1000, seed: int = 0) -> dict[str, float]:
    """CI of objective(a); with b (same units, same order): paired test of objective(a) - objective(b)."""
    C = len(spec.labels)
    A = _cms(a, C)
    B = _cms(b, C) if b is not None else None
    if B is not None and len(B) != len(A):
        raise ValueError("paired bootstrap needs the same units")
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        idx = rng.integers(0, len(A), len(A))
        va = objective_from_cm(spec, A[idx].sum(0))
        vals.append(va - objective_from_cm(spec, B[idx].sum(0)) if B is not None else va)
    v = np.asarray(vals)
    lo, hi = np.quantile(v, [0.025, 0.975])
    out = {"lo": float(lo), "hi": float(hi)}
    if B is not None:
        out.update(delta=objective_from_cm(spec, A.sum(0)) - objective_from_cm(spec, B.sum(0)), p_better=float((v > 0).mean() + 0.5 * (v == 0).mean()))
    return out
