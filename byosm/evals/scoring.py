from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from .. import metrics
from ..modalities import get_modality
from ..predictors import Predictor
from ..rt import hysteresis
from ..spec import TaskSpec
from .grouped import grouped_metrics

BUSY_LOAD = 1.0
FramePairs = list[tuple[np.ndarray, np.ndarray]]
ACTIVATION_GRID = np.linspace(0.1, 0.9, 33)
CI_RESAMPLES = 500


def _collect(
    pred: Predictor, spec: TaskSpec, rows: list[dict], root: Path
) -> tuple[np.ndarray, np.ndarray, FramePairs]:
    mod = get_modality(spec, str(root))
    if spec.task_type == "frame":
        per = []
        listed = pred.frame_rows(rows, mod) if getattr(pred, "wants_rows", False) else None
        for k, r in enumerate(rows):
            pr = listed[k] if listed is not None else pred.frame_probs(mod.load(r))
            y = np.asarray(mod.frame_labels(r))
            n = min(len(pr), len(y))
            per.append((pr[:n], y[:n]))
        return np.concatenate([a for a, _ in per]), np.concatenate([b for _, b in per]), per
    idx = spec.label_index
    y = np.array([idx[r["label"]] for r in rows])
    return _within(row_probs(pred, rows, mod), spec, pred.label_space), y, []


def row_probs(pred: Predictor, rows: list[dict], mod: Any) -> np.ndarray:
    return pred.probs_rows(rows, mod) if getattr(pred, "wants_rows", False) else pred.probs_batch(mod.load_many(rows))


def _within(probs: np.ndarray, spec: TaskSpec, space: list[str] | None) -> np.ndarray:
    if not space:
        return probs
    out = np.array(probs, np.float32)
    out[:, [i for i, n in enumerate(spec.label_names) if n not in space]] = 0
    return out


def _hyst(spec: TaskSpec, per: list, act: float) -> tuple[np.ndarray, np.ndarray]:
    d, hop = spec.decision, spec.frame_hop_ms / 1000
    dec = np.concatenate(
        [
            hysteresis(
                p[:, spec.pos_index],
                hop,
                act,
                d.deactivation_offset,
                d.min_on_ms,
                d.min_off_ms,
                d.smooth_alpha_per_32ms,
            )
            for p, _ in per
        ]
    )
    y = np.concatenate([y for _, y in per])
    return dec, y


def _onehot(spec: TaskSpec, dec: np.ndarray) -> np.ndarray:
    neg = next(i for i in range(len(spec.labels)) if i != spec.pos_index)
    out = np.zeros((len(dec), len(spec.labels)), np.float32)
    out[np.arange(len(dec)), np.where(dec, spec.pos_index, neg)] = 1
    return out


def _fit_activation(spec: TaskSpec, per: FramePairs) -> float:
    d = spec.decision
    if d.threshold is not None or not per:
        return 0.5 if d.threshold is None else d.threshold
    ycal = np.concatenate([y for _, y in per]) == spec.pos_index

    def f1_at(a: float) -> float:
        return metrics._prf_at(ycal, _hyst(spec, per, a)[0].astype(float), 0.5)[2]

    return float(max(ACTIVATION_GRID, key=f1_at))


def _cross_fit_frames(spec: TaskSpec, perc: FramePairs) -> np.ndarray:
    folds = [perc[0::2], perc[1::2]] if len(perc) > 1 else [perc, perc]
    acts = [_fit_activation(spec, folds[1]), _fit_activation(spec, folds[0])]
    dec = np.concatenate([_hyst(spec, [f], acts[i % 2])[0] for i, f in enumerate(perc)])
    return _onehot(spec, dec).argmax(-1)


def _decide_frames(spec: TaskSpec, yt: np.ndarray, perc: FramePairs, pert: FramePairs) -> tuple:
    act = _fit_activation(spec, perc)
    dec, _ = _hyst(spec, pert, act)
    m = metrics.compute(spec, yt, _onehot(spec, dec), 0.5)
    m["threshold"] = act
    dec_c = _cross_fit_frames(spec, perc) if perc else None
    return m, _onehot(spec, dec).argmax(-1), dec_c


def _cross_fit_rows(spec: TaskSpec, yc: np.ndarray, pc: np.ndarray, thr: float | None) -> np.ndarray:
    dec_c = metrics.predict(spec, pc, thr)
    if spec.decision.threshold is None and metrics._binary(spec) is not None and len(pc) > 1:
        for a, b in ((0, 1), (1, 0)):
            dec_c[a::2] = metrics.predict(spec, pc[a::2], metrics.calibrate(spec, yc[b::2], pc[b::2]))
    return dec_c


def _decide_rows(spec: TaskSpec, yt: np.ndarray, pt: np.ndarray, yc: Any, pc: Any, lengths: Any = None) -> tuple:
    thr = spec.decision.threshold
    if thr is None and pc is not None:
        thr = metrics.calibrate(spec, yc, pc)
    m = metrics.compute(spec, yt, pt, thr, lengths)
    dec_c = _cross_fit_rows(spec, yc, pc, thr) if pc is not None else None
    return m, metrics.predict(spec, pt, thr), dec_c


def _groups(per: FramePairs) -> list[int] | None:
    return [len(y) for _, y in per] if per else None


def _add_threshold_free(m: dict[str, Any], spec: TaskSpec, yt: np.ndarray, pt: np.ndarray) -> None:
    binary = yt == spec.pos_index
    if metrics._binary(spec) is not None and 0 < binary.sum() < len(binary):
        m["roc_auc"] = float(roc_auc_score(binary, pt[:, spec.pos_index]))
        m["avg_precision"] = float(average_precision_score(binary, pt[:, spec.pos_index]))


def busy(result: dict[str, Any]) -> bool:
    load = result.get("latency_load")
    return load is not None and load > BUSY_LOAD


def fair_ms(result: dict[str, Any]) -> float | None:
    return None if busy(result) else result.get("latency_ms")


def add_speed(out: dict[str, Any], pred: Predictor, spec: TaskSpec, test: list[dict], root: Path) -> None:
    ck = pred.stream_chunk_ms()
    if ck is not None:
        out["stream_ms_per_chunk"], out["chunk_ms"] = round(ck, 4), pred.chunk_ms
        return
    mod = get_modality(spec, str(root))
    timed = (spec.task_type == "classify" or spec.modality == "text") and not getattr(pred, "wants_rows", False)
    lat = pred.latency_ms(mod.load(test[0])) if timed else None
    out["latency_ms"] = round(lat, 3) if lat is not None and not math.isnan(lat) else None
    if out["latency_ms"] is not None:
        from ..machine import load_per_core

        out["latency_load"] = load_per_core()


def score(
    pred: Predictor, spec: TaskSpec, calib: list[dict], test: list[dict], root: Path, space: list[str] | None = None
) -> dict[str, Any]:
    t0 = time.time()
    pred.label_space = space if space and space != spec.label_names else None
    pc, yc, perc = _collect(pred, spec, calib, root) if calib else (None, None, [])
    pt, yt, pert = _collect(pred, spec, test, root)
    out: dict[str, Any] = {"predictor": pred.id, "kind": pred.kind, "params": pred.params}
    n_labels = len(spec.labels)
    if spec.decision.kind == "hysteresis" and spec.task_type == "frame":
        m, dec_t, dec_c = _decide_frames(spec, yt, perc, pert)
    else:
        m, dec_t, dec_c = _decide_rows(spec, yt, pt, yc, pc, _groups(pert))
    names = spec.label_names if metrics.is_span(spec) else None
    out["units"] = {"test": metrics.units(n_labels, yt, dec_t, _groups(pert), names)}
    m["ci95"] = metrics.bootstrap(spec, out["units"]["test"], n=CI_RESAMPLES)
    if dec_c is not None:
        out["units"]["calib"] = metrics.units(n_labels, yc, dec_c, _groups(perc), names)
        out["calib_objective"] = metrics.objective_of_units(spec, out["units"]["calib"])
    _add_threshold_free(m, spec, yt, pt)
    m.update(grouped_metrics(spec, calib, test, pc, pt, dec_t))
    out.update(
        metrics=m, objective=m["objective"], n_test=len(test), n_calib=len(calib), seconds=round(time.time() - t0, 1)
    )
    add_speed(out, pred, spec, test, root)
    return out
