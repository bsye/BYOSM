from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ..config import Project
from ..predictors import ExperimentPredictor, Predictor, get_baseline
from ..spec import EvalSuite, TaskSpec
from .dumps import _dumps
from .scoring import score
from .suites import fingerprint, materialize

Job = tuple[str, str, Callable[[], Predictor]]
SuiteData = tuple[list[dict], list[dict], Path]


def _cache(p: Project, pred_kind: str, pid: str, suite: str) -> Path:
    if pred_kind == "experiment":
        base = p.exp_dir(pid) / "results"
    else:
        base = p.baselines_dir / pid.replace("/", "_").replace(":", "_")
    return base / f"{suite}.json"


def _kept_results(p: Project, base: TaskSpec) -> dict[str, dict[str, dict]]:
    keep_ids, keep_suites = set(p.experiments()) | set(base.baselines), {s.name for s in base.evals}
    old = p.board()
    return {pid: {s: r for s, r in by.items() if s in keep_suites} for pid, by in old.items() if pid in keep_ids}


def _trained_on_split(p: Project, exp: str) -> bool:
    meta_f = p.exp_dir(exp) / "meta.json"
    return bool(meta_f.exists() and json.loads(meta_f.read_text()).get("n_test_folded"))


def _experiment_predictor(p: Project, base: TaskSpec, exp: str) -> ExperimentPredictor:
    spec = TaskSpec.load(p.exp_dir(exp) / "spec.yaml").model_copy(update={"decision": base.decision})
    return ExperimentPredictor(spec, p.exp_dir(exp), exp)


def _jobs(p: Project, base: TaskSpec, suite: EvalSuite, exps: list[str], bls: list[str]) -> list[Job]:
    jobs: list[Job] = []
    for e in exps:
        if suite.kind == "builtin" and _trained_on_split(p, e):
            continue
        jobs.append(("experiment", e, lambda e=e: _experiment_predictor(p, base, e)))
    for b in bls:
        jobs.append(("baseline", b, lambda b=b: get_baseline(b, base)))
    return jobs


def _cached_or_scored(
    p: Project, base: TaskSpec, suite: str, job: Job, fp: str, force: bool, data: SuiteData, log: Callable[[str], None]
) -> dict | None:
    kind, pid, make = job
    calib, test, root = data
    cache = _cache(p, kind, pid, suite)
    if not force and cache.exists() and json.loads(cache.read_text()).get("fingerprint") == fp:
        return json.loads(cache.read_text())
    log(f"  scoring {pid} on {suite} ({len(test)} test items)...")
    try:
        r = score(make(), base, calib, test, root)
    except Exception as ex:
        log(f"  {pid} failed on {suite}: {type(ex).__name__}: {ex}")
        return None
    r["fingerprint"] = fp
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(_dumps(r))
    return r


def run(
    name: str,
    experiments: list[str] | None = None,
    baselines: list[str] | None = None,
    suites: list[str] | None = None,
    force: bool = False,
    log: Callable[[str], None] = print,
) -> dict[str, dict[str, dict]]:
    p = Project(name)
    base = TaskSpec.load(p.spec_path)
    exps = experiments if experiments is not None else p.experiments()
    bls = baselines if baselines is not None else base.baselines
    results = _kept_results(p, base)
    for suite in base.evals:
        if suites and suite.name not in suites:
            continue
        data = materialize(p, base, suite, log)
        if not data[1]:
            log(f"  suite {suite.name}: empty, skipped")
            continue
        fp = fingerprint(p, base, suite)
        for job in _jobs(p, base, suite, exps, bls):
            r = _cached_or_scored(p, base, suite.name, job, fp, force, data, log)
            if r is not None:
                results.setdefault(job[1], {})[suite.name] = r
    p.board_path.write_text(_dumps(results))
    return results


def _board_row(pid: str, by_suite: dict[str, dict], suites: list[str], promoted: bool) -> dict[str, Any]:
    any_r = next(iter(by_suite.values()))
    row: dict[str, Any] = {
        "id": pid,
        "kind": any_r["kind"],
        "params": any_r.get("params"),
        "promoted": promoted,
        "stream_ms": any_r.get("stream_ms_per_chunk"),
        "chunk_ms": any_r.get("chunk_ms"),
        "latency_ms": any_r.get("latency_ms"),
    }
    row.update({s: by_suite.get(s, {}).get("objective") for s in suites})
    row.update({f"{s}:auc": by_suite.get(s, {}).get("metrics", {}).get("roc_auc") for s in suites})
    row.update({f"{s}:calib": by_suite.get(s, {}).get("calib_objective") for s in suites})
    row.update({f"{s}:ci": by_suite.get(s, {}).get("metrics", {}).get("ci95") for s in suites})
    for s in suites:
        m = by_suite[s]["metrics"] if s in by_suite else None
        row[f"{s}:p/r"] = (m.get("precision"), m.get("recall")) if m is not None else None
    return row


def board_rows(name: str) -> tuple[list[str], list[dict]]:
    p = Project(name)
    base = TaskSpec.load(p.spec_path)
    suites = [s.name for s in base.evals]
    rows = [_board_row(pid, by_suite, suites, pid == p.promoted) for pid, by_suite in p.board().items()]
    prim = base.primary_suite()
    rows.sort(key=lambda r: -(r.get(prim) or -1))
    return suites, rows
