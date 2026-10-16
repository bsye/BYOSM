"""Eval suites + scoring + the board.

Every predictor (experiment or baseline) is scored the same way on every suite:
  1. probabilities on the suite's calib split -> calibrate the decision policy (threshold / hysteresis activation)
  2. probabilities on the suite's test split  -> objective + precision/recall/F1 + threshold-free ROC-AUC / AP
Results are cached per (predictor, suite fingerprint); the board is a table of cached results.
"""
from __future__ import annotations

import hashlib
import json
import random
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from . import metrics
from .config import Project
from .data.store import Dataset, read, record, write
from .modalities import get_modality
from .predictors import ExperimentPredictor, Predictor, get_baseline
from .rt import hysteresis
from .spec import EvalSuite, TaskSpec


# --------------------------------------------------------------------------------------------- suites


def suite_dir(p: Project, suite: EvalSuite) -> Path:
    return p.evals_dir / suite.name


def fingerprint(p: Project, spec: TaskSpec, suite: EvalSuite) -> str:
    d = suite_dir(p, suite)
    blob = suite.model_dump_json() + "".join(f.read_text()[:100000] for f in sorted(d.glob("*.jsonl"))) if d.exists() else suite.model_dump_json()
    if suite.kind == "builtin":
        blob += spec.data_key()
    blob += spec.decision.model_dump_json()
    return hashlib.sha1(blob.encode()).hexdigest()[:10]


def materialize(p: Project, base: TaskSpec, suite: EvalSuite, log: Callable[[str], None] = print) -> tuple[list[dict], list[dict], Path]:
    """-> (calib_rows, test_rows, media_root). Non-builtin suites are frozen to evals/<suite>/ on first use."""
    if suite.kind == "builtin":
        dd = p.data_dir_for(base)
        ds = Dataset(dd)
        return ds.load("dev"), ds.load("test"), dd
    d = suite_dir(p, suite)
    if (d / "test.jsonl").exists():
        return read(d / "calib.jsonl"), read(d / "test.jsonl"), d
    d.mkdir(parents=True, exist_ok=True)
    calib, test = [], []
    if suite.kind == "local":
        for r in read(Path(suite.path)):
            (calib if r.get("split") == "calib" else test).append(r)
    elif suite.kind == "hf" and suite.source and suite.source.format == "segment_files":
        from .data import segments
        from .modalities.audio import load_wav

        src = suite.source
        hop = int(base.data.audio.sample_rate * base.frame_hop_ms / 1000)
        n_cal = max(1, src.max_files // 5)
        for part, n, rows in (("calib", n_cal, calib), ("test", src.max_files, test)):
            for f in segments.fetch(src, (part,), n, p.root / "cache"):
                n_frames = len(load_wav(f, base.data.audio.sample_rate)) // hop
                rows.append(record(str(f), segments.frame_labels(f, n_frames, base).tolist(), f"hf:{src.dataset}", file=f.name))
        log(f"  suite {suite.name}: {len(calib)} calib + {len(test)} test files from {src.dataset}")
    elif suite.kind == "hf" and suite.source:
        from .data import _save_media, hf_source

        tmp = base.model_copy(deep=True)
        tmp.data.hf_sources = [suite.source]
        src = (hf_source.resolve(tmp, None, log) or [suite.source])[0]
        rows = [(x, y) for x, y in hf_source.take(src, base, suite.max_items * 2) if y]
        random.Random(0).shuffle(rows)
        for i, (x, y) in enumerate(rows[: suite.max_items]):
            inp = x if base.modality == "text" else _save_media(base, d, "media", x)
            (calib if i % 5 == 0 else test).append(record(inp, y, f"hf:{src.dataset}"))
        suite = suite.model_copy(update={"source": src})
    else:
        raise ValueError(f"suite {suite.name}: unsupported kind/source")
    write(d / "calib.jsonl", calib)
    write(d / "test.jsonl", test)
    (d / "suite.json").write_text(suite.model_dump_json(indent=2))
    return calib, test, d


# --------------------------------------------------------------------------------------------- scoring


def _collect(pred: Predictor, spec: TaskSpec, rows: list[dict], root: Path) -> tuple[np.ndarray, np.ndarray, list[tuple[np.ndarray, np.ndarray]]]:
    mod = get_modality(spec, str(root))
    if spec.task_type == "frame":
        per = []
        for r in rows:
            raw = mod.load(r)
            pr = pred.frame_probs(raw)
            y = np.asarray(mod.frame_labels(r))
            n = min(len(pr), len(y))
            per.append((pr[:n], y[:n]))
        return np.concatenate([a for a, _ in per]), np.concatenate([b for _, b in per]), per
    raws = [mod.load(r) for r in rows]
    idx = {n: i for i, n in enumerate(spec.label_names)}
    return pred.probs_batch(raws), np.array([idx[r["label"]] for r in rows]), []


def _hyst(spec: TaskSpec, per: list, act: float) -> tuple[np.ndarray, np.ndarray]:
    d, hop = spec.decision, spec.frame_hop_ms / 1000
    dec = np.concatenate([hysteresis(p[:, spec.pos_index], hop, act, d.deactivation_offset, d.min_on_ms, d.min_off_ms, d.smooth_alpha_per_32ms) for p, _ in per])
    y = np.concatenate([y for _, y in per])
    return dec, y


def _onehot(spec: TaskSpec, dec: np.ndarray) -> np.ndarray:
    neg = next(i for i in range(len(spec.labels)) if i != spec.pos_index)
    out = np.zeros((len(dec), len(spec.labels)), np.float32)
    out[np.arange(len(dec)), np.where(dec, spec.pos_index, neg)] = 1
    return out


def score(pred: Predictor, spec: TaskSpec, calib: list[dict], test: list[dict], root: Path) -> dict[str, Any]:
    t0 = time.time()
    pc, yc, perc = _collect(pred, spec, calib, root) if calib else (None, None, [])
    pt, yt, pert = _collect(pred, spec, test, root)
    d = spec.decision
    out: dict[str, Any] = {"predictor": pred.id, "kind": pred.kind, "params": pred.params}
    if d.kind == "hysteresis" and spec.task_type == "frame":
        act = d.threshold
        if act is None and perc:
            ycal = np.concatenate([y for _, y in perc]) == spec.pos_index
            act = float(max(np.linspace(0.1, 0.9, 33), key=lambda a: metrics._prf_at(ycal, _hyst(spec, perc, a)[0].astype(float), 0.5)[2]))
        act = 0.5 if act is None else act
        dec, _ = _hyst(spec, pert, act)
        m = metrics.compute(spec, yt, _onehot(spec, dec), 0.5)
        m["threshold"] = act
    else:
        thr = d.threshold if d.threshold is not None else (metrics.calibrate(spec, yc, pc) if pc is not None else None)
        m = metrics.compute(spec, yt, pt, thr)
    binary = yt == spec.pos_index
    if 0 < binary.sum() < len(binary):
        m["roc_auc"] = float(roc_auc_score(binary, pt[:, spec.pos_index]))
        m["avg_precision"] = float(average_precision_score(binary, pt[:, spec.pos_index]))
    m.pop("confusion", None)
    out.update(metrics=m, objective=m["objective"], n_test=len(test), n_calib=len(calib), seconds=round(time.time() - t0, 1))
    ck = pred.stream_chunk_ms()
    if ck is not None:
        out["stream_ms_per_chunk"], out["chunk_ms"] = round(ck, 4), pred.chunk_ms
    else:
        mod = get_modality(spec, str(root))
        out["latency_ms"] = round(pred.latency_ms(mod.load(test[0])), 3) if spec.task_type == "classify" else None
    return out


# --------------------------------------------------------------------------------------------- run + board


def _cache(p: Project, pred_kind: str, pid: str, suite: str) -> Path:
    base = p.exp_dir(pid) / "results" if pred_kind == "experiment" else p.baselines_dir / pid.replace("/", "_").replace(":", "_")
    return base / f"{suite}.json"


def run(name: str, experiments: list[str] | None = None, baselines: list[str] | None = None, suites: list[str] | None = None,
        force: bool = False, log: Callable[[str], None] = print) -> dict[str, dict[str, dict]]:
    p = Project(name)
    base = TaskSpec.load(p.spec_path)
    exps = experiments if experiments is not None else p.experiments()
    bls = baselines if baselines is not None else base.baselines
    results: dict[str, dict[str, dict]] = {}
    for suite in base.evals:
        if suites and suite.name not in suites:
            continue
        calib, test, root = materialize(p, base, suite, log)
        if not test:
            log(f"  suite {suite.name}: empty, skipped")
            continue
        fp = fingerprint(p, base, suite)
        jobs: list[tuple[str, str, Callable[[], Predictor]]] = []
        for e in exps:
            jobs.append(("experiment", e, lambda e=e: ExperimentPredictor(TaskSpec.load(p.exp_dir(e) / "spec.yaml").model_copy(update={"decision": base.decision}), p.exp_dir(e), e)))
        for b in bls:
            jobs.append(("baseline", b, lambda b=b: get_baseline(b, base)))
        for kind, pid, make in jobs:
            cache = _cache(p, kind, pid, suite.name)
            if not force and cache.exists() and json.loads(cache.read_text()).get("fingerprint") == fp:
                r = json.loads(cache.read_text())
            else:
                log(f"  scoring {pid} on {suite.name} ({len(test)} test items)...")
                try:
                    r = score(make(), base, calib, test, root)
                except Exception as ex:  # noqa: BLE001
                    log(f"  {pid} failed on {suite.name}: {type(ex).__name__}: {ex}")
                    continue
                r["fingerprint"] = fp
                cache.parent.mkdir(parents=True, exist_ok=True)
                cache.write_text(json.dumps(r, indent=2))
            results.setdefault(pid, {})[suite.name] = r
    (p.root / "board.json").write_text(json.dumps(results, indent=2))
    return results


def board_rows(name: str) -> tuple[list[str], list[dict]]:
    p = Project(name)
    base = TaskSpec.load(p.spec_path)
    res = json.loads((p.root / "board.json").read_text()) if (p.root / "board.json").exists() else {}
    suites = [s.name for s in base.evals]
    rows = []
    for pid, by_suite in res.items():
        any_r = next(iter(by_suite.values()))
        rows.append({"id": pid, "kind": any_r["kind"], "params": any_r.get("params"), "promoted": pid == p.promoted,
                     "stream_ms": any_r.get("stream_ms_per_chunk"), "chunk_ms": any_r.get("chunk_ms"), "latency_ms": any_r.get("latency_ms"),
                     **{s: by_suite.get(s, {}).get("objective") for s in suites},
                     **{f"{s}:auc": by_suite.get(s, {}).get("metrics", {}).get("roc_auc") for s in suites},
                     **{f"{s}:p/r": (by_suite[s]["metrics"].get("precision"), by_suite[s]["metrics"].get("recall")) if s in by_suite else None for s in suites}})
    prim = base.primary_suite()
    rows.sort(key=lambda r: -(r.get(prim) or -1))
    return suites, rows
