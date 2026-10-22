from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support, roc_auc_score

from .resample import bootstrap_stats
from .rt import spans
from .spec import Objective, TaskSpec


def is_span(spec: TaskSpec) -> bool:
    return spec.objective.metric == "span_f1"


def span_counts(names: list[str], y: np.ndarray, pred: np.ndarray, lengths: list[int]) -> np.ndarray:
    out, s = [], 0
    for n in lengths:
        gold = set(spans([names[i] for i in y[s : s + n]]))
        found = set(spans([names[i] for i in pred[s : s + n]]))
        out.append([len(gold & found), len(found - gold), len(gold - found)])
        s += n
    return np.asarray(out, np.int64).reshape(-1, 3)


def f1_of_counts(counts: np.ndarray) -> float:
    tp, fp, fn = counts.astype(float)
    return 2 * tp / max(1.0, 2 * tp + fp + fn)


def _binary(spec: TaskSpec) -> int | None:
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


def _threshold_value(metric: str, constraint: float, p: float, r: float, f: float) -> float:
    if metric == "recall_at_precision":
        return r if p >= constraint else -1 + p
    if metric == "precision_at_recall":
        return p if r >= constraint else -1 + r
    return f


def calibrate(spec: TaskSpec, y: np.ndarray, probs: np.ndarray) -> float | None:
    pos = _binary(spec)
    if pos is None:
        return None
    yb, score = y == pos, probs[:, pos]
    cands = np.unique(np.concatenate([np.quantile(score, np.linspace(0, 1, 401)), [0.5]]))
    best, best_thr = -1.0, 0.5
    m, c = spec.objective.metric, spec.objective.constraint
    for thr in cands:
        v = _threshold_value(m, c, *_prf_at(yb, score, thr))
        if v > best:
            best, best_thr = v, float(thr)
    return best_thr


def compute(
    spec: TaskSpec, y: np.ndarray, probs: np.ndarray, threshold: float | None, lengths: list[int] | None = None
) -> dict[str, Any]:
    names = spec.label_names
    pred = predict(spec, probs, threshold)
    labels = list(range(len(names)))
    p, r, f, s = precision_recall_fscore_support(y, pred, labels=labels, zero_division=0)
    out: dict[str, Any] = {
        "n": int(len(y)),
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, labels=labels, average="macro", zero_division=0)),
        "per_label": {
            names[i]: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]), "support": int(s[i])}
            for i in labels
        },
        "confusion": confusion_matrix(y, pred, labels=labels).tolist(),
        "threshold": threshold,
    }
    pos = _binary(spec)
    if pos is not None:
        yb = y == pos
        out.update(precision=float(p[pos]), recall=float(r[pos]), f1=float(f[pos]))
        if 0 < yb.sum() < len(yb):
            out["roc_auc"] = float(roc_auc_score(yb, probs[:, pos]))
    if is_span(spec):
        tp, fp, fn = span_counts(names, y, pred, lengths or [len(y)]).sum(0).astype(float)
        out.update(span_precision=tp / max(1.0, tp + fp), span_recall=tp / max(1.0, tp + fn))
        out["span_f1"] = f1_of_counts(np.array([tp, fp, fn]))
    out["objective"] = objective_value(spec, out)
    return out


CONSTRAINED = {"recall_at_precision": ("recall", "precision"), "precision_at_recall": ("precision", "recall")}


def metric_label(o: Objective) -> str:
    if o.metric in CONSTRAINED:
        found, bound = CONSTRAINED[o.metric]
        return f"{found} at {o.constraint:.0%} {bound}"
    return o.metric.replace("_", " ").replace("f1", "F1")


def unreached(o: Objective, m: dict[str, Any]) -> float | None:
    if o.metric not in CONSTRAINED:
        return None
    reached = m.get(CONSTRAINED[o.metric][1])
    return reached if reached is not None and reached < o.constraint else None


def objective_value(spec: TaskSpec, m: dict[str, Any]) -> float:
    o = spec.objective
    if o.metric == "recall_at_precision":
        return m.get("recall", 0.0) if m.get("precision", 0.0) >= o.constraint else 0.0
    if o.metric == "precision_at_recall":
        return m.get("precision", 0.0) if m.get("recall", 0.0) >= o.constraint else 0.0
    if o.metric == "f1":
        return m.get("f1", m["macro_f1"])
    return float(m[o.metric])


def units(
    n_labels: int, y: np.ndarray, pred: np.ndarray, groups: list[int] | None = None, names: list[str] | None = None
) -> dict[str, Any]:
    if names is not None:
        return {"spans": span_counts(names, np.asarray(y), np.asarray(pred), groups or [len(y)]).tolist()}
    if groups is None:
        return {"y": np.asarray(y).astype(int).tolist(), "pred": np.asarray(pred).astype(int).tolist()}
    cms, s = [], 0
    for n in groups:
        cms.append(confusion_matrix(y[s : s + n], pred[s : s + n], labels=list(range(n_labels))).ravel().tolist())
        s += n
    return {"cm": cms}


def _cms(u: dict[str, Any], n_labels: int) -> np.ndarray:
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


def _unit_stats(spec: TaskSpec, u: dict[str, Any]) -> np.ndarray:
    if "spans" in u:
        return np.asarray(u["spans"], np.int64).reshape(-1, 3)
    return _cms(u, len(spec.labels))


def _objective_of_stats(spec: TaskSpec, summed: np.ndarray) -> float:
    return f1_of_counts(summed) if summed.ndim == 1 else objective_from_cm(spec, summed)


def objective_of_units(spec: TaskSpec, u: dict[str, Any]) -> float:
    return _objective_of_stats(spec, _unit_stats(spec, u).sum(0))


def bootstrap(
    spec: TaskSpec, a: dict[str, Any], b: dict[str, Any] | None = None, n: int = 1000, seed: int = 0
) -> dict[str, float]:
    sb = _unit_stats(spec, b) if b is not None else None
    return bootstrap_stats(_unit_stats(spec, a), sb, lambda summed: _objective_of_stats(spec, summed), n, seed)
