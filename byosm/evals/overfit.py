from __future__ import annotations

import hashlib
import json
import random
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .. import metrics
from ..data.store import Dataset, ShingleIndex
from ..events import stage
from ..improve.diagnosis import scored_experiment, training_summary
from ..project import Project
from ..spec import EvalSuite, TaskSpec
from ..train.loop import split_rows
from . import overfit_checks as checks
from .board import experiment_predictor
from .dumps import _dumps
from .scoring import CI_RESAMPLES, score
from .selection import selection_value
from .suites import fingerprint, stored

TRAIN_SAMPLE = 1000
SAMPLE_SEED = 0
OVERFIT_VERSION = "1"
OVERFIT = "overfit"
VERDICTS = {
    "ok": "No sign of overfitting or leakage.",
    "warn": "Some signs of overfitting: read the findings.",
    "fail": "Overfitting or leakage found.",
}


def _hash(*parts: str) -> str:
    return hashlib.sha1("\n".join(parts).encode()).hexdigest()[:10]


def _primary(base: TaskSpec) -> EvalSuite:
    return next(s for s in base.evals if s.name == base.primary_suite())


def _fingerprints(p: Project, base: TaskSpec, exp: str) -> tuple[str, str]:
    meta = json.dumps(p.meta(exp), sort_keys=True)
    train_fp = _hash(exp, meta, fingerprint(p, base, _primary(base)), str(TRAIN_SAMPLE), OVERFIT_VERSION)
    return _hash(train_fp, json.dumps(p.board(), sort_keys=True)), train_fp


def _train_rows(p: Project, spec: TaskSpec) -> tuple[list[dict], Path]:
    dd = p.data_dir_for(spec)
    rows, _, _ = split_rows(spec, Dataset(dd), random.Random(spec.train.seed), lambda _: None)
    return rows, dd


def _absolute(spec: TaskSpec, rows: list[dict], dd: Path) -> list[dict]:
    return rows if spec.modality == "text" else [{**r, "input": str(dd / r["input"])} for r in rows]


def _train_score(
    p: Project, base: TaskSpec, exp: str, spec: TaskSpec, rows: tuple[list[dict], Path], log: Callable[[str], None]
) -> dict[str, Any]:
    train_rows, dd = rows
    suite = _primary(base)
    calib, _, root = stored(p, base, suite) or ([], [], dd)
    sample = random.Random(SAMPLE_SEED).sample(train_rows, min(TRAIN_SAMPLE, len(train_rows)))
    log(f"  scoring {exp} on {len(sample)} of its {len(train_rows)} training rows...")
    t0 = time.time()
    space = suite.label_space(base.label_names)
    r = score(experiment_predictor(p, base, exp), base, calib, _absolute(spec, sample, dd), root, space)
    ci = r["metrics"]["ci95"]
    return {"objective": r["objective"], "ci95": ci, "n": len(sample), "seconds": round(time.time() - t0, 1)}


def _calib_ci(base: TaskSpec, r: dict) -> dict | None:
    u = r.get("units", {}).get("calib")
    return metrics.bootstrap(base, u, n=CI_RESAMPLES) if u else None


def _drifts(base: TaskSpec, results: dict[str, dict]) -> list[dict]:
    out = []
    for s, r in results.items():
        ci = _calib_ci(base, r)
        if ci is not None and r.get("calib_objective") is not None:
            out.append(checks.drift(s, r["calib_objective"], ci, r["objective"], r["metrics"]["ci95"]))
    return out


def _overlaps(p: Project, base: TaskSpec, spec: TaskSpec, train_rows: list[dict], results: dict) -> list[dict]:
    if spec.modality != "text":
        return []
    index = ShingleIndex(train_rows)
    kept = {s.name: stored(p, base, s) for s in base.evals if s.name in results}
    units = {s: results[s].get("units", {}).get("test", {}) for s in kept}
    return [checks.overlap(s, index, k[1], base, units[s]) for s, k in kept.items() if k and k[1]]


def _selection(base: TaskSpec, board: dict, exp: str, prim: str) -> list[dict]:
    ci = _calib_ci(base, board[exp][prim])
    mine = selection_value(board[exp][prim])
    if ci is None or mine is None:
        return []
    exps = {pid: by[prim] for pid, by in board.items() if prim in by and by[prim]["kind"] == "experiment"}
    others = [v for pid, r in exps.items() if pid != exp and (v := selection_value(r)) is not None]
    return [checks.selection(mine, others, ci)]


def _curve(p: Project, exp: str, spec: TaskSpec) -> dict:
    t = training_summary(p.meta(exp), spec)
    return checks.curve(t["curve"], t["losses"], t["best_epoch"], t["epochs_cfg"])


def _report(p: Project, base: TaskSpec, exp: str, train: dict, train_rows: list[dict]) -> dict[str, Any]:
    spec = p.exp_spec(exp)
    board, prim = p.board(), base.primary_suite()
    results = board[exp]
    r = results[prim]
    scores = {"train": train["objective"], "dev": p.meta(exp).get("dev", {}).get("objective")}
    scores |= {"calib": r.get("calib_objective"), "test": r["objective"]}
    findings = [checks.gap(scores["train"], scores["dev"], scores["calib"], scores["test"])]
    findings += _drifts(base, results) + _overlaps(p, base, spec, train_rows, results)
    findings += _selection(base, board, exp, prim) + [_curve(p, exp, spec)]
    verdict = checks.worst(findings)
    skipped = [] if spec.modality == "text" else ["train/test overlap, because only text inputs are compared"]
    head = {"specialist": p.name, "experiment": exp, "suite": prim, "metric": base.objective.metric}
    verdicts = {"verdict": verdict, "summary": VERDICTS[verdict], "scores": scores, "train": train}
    return {**head, **verdicts, "findings": findings, "skipped": skipped}


@stage("Overfitting check")
def check(
    name: str,
    exp: str | None = None,
    ws: Path | None = None,
    force: bool = False,
    log: Callable[[str], None] = print,
) -> dict[str, Any]:
    p = Project(name, ws)
    base = p.spec()
    board = p.board()
    exp = exp or scored_experiment(p, board)
    if base.primary_suite() not in board.get(exp, {}):
        raise RuntimeError(f"{exp} has no {base.primary_suite()} result yet - run `byosm eval {name}` first")
    old = ({} if force else p.report(OVERFIT)) or {}
    fp, train_fp = _fingerprints(p, base, exp)
    if old.get("fingerprint") == fp:
        return old
    spec = p.exp_spec(exp)
    rows = _train_rows(p, spec)
    train = old["train"] if old.get("train_fingerprint") == train_fp else _train_score(p, base, exp, spec, rows, log)
    report = {"fingerprint": fp, "train_fingerprint": train_fp, **_report(p, base, exp, train, rows[0])}
    p.save_report(OVERFIT, json.loads(_dumps(report)))
    return report
