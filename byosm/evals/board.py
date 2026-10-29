from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from ..events import emit, stage
from ..predictors import ExperimentPredictor, Predictor, get_baseline
from ..project import Project
from ..spec import EvalSuite, TaskSpec
from .grouped import summary
from .identity import baseline_identity, experiment_identity
from .scoring import score
from .suites import fingerprint, materialize

Job = tuple[str, str, Callable[[], Predictor]]
SuiteData = tuple[list[dict], list[dict], Path, list[str]]


def experiment_predictor(p: Project, base: TaskSpec, exp: str) -> ExperimentPredictor:
    spec = p.exp_spec(exp).model_copy(update={"decision": base.decision})
    return ExperimentPredictor(spec, p.meta(exp), p.exp_dir(exp), exp)


def _jobs(p: Project, base: TaskSpec, suite: EvalSuite, exps: list[str], bls: list[str]) -> list[Job]:
    jobs: list[Job] = []
    for e in exps:
        if suite.kind == "builtin" and p.meta(e).get("n_test_folded"):
            continue
        jobs.append(("experiment", e, lambda e=e: experiment_predictor(p, base, e)))
    for b in bls:
        jobs.append(("baseline", b, lambda b=b: get_baseline(b, base)))
    return jobs


def _identity(p: Project, kind: str, pid: str) -> str | None:
    return experiment_identity(p.meta(pid)) if kind == "experiment" else baseline_identity(pid)


def _fresh(p: Project, pid: str, suite: str, cached: dict, fp: str, kind: str) -> bool:
    if cached.get("fingerprint") != fp:
        return False
    identity = _identity(p, kind, pid)
    if identity is not None and "identity" not in cached:
        cached["identity"] = identity
        p.save_result(pid, suite, cached)
    return cached.get("identity") == identity


def _cached_or_scored(
    p: Project, base: TaskSpec, suite: str, job: Job, fp: str, force: bool, data: SuiteData, log: Callable[[str], None]
) -> dict | None:
    kind, pid, make = job
    calib, test, root, space = data
    cached = (None if force else p.result(pid, suite)) or {}
    if _fresh(p, pid, suite, cached, fp, kind):
        return cached
    log(f"  scoring {pid} on {suite} ({len(test)} test items)...")
    try:
        r = score(make(), base, calib, test, root, space, suite)
        identity = _identity(p, kind, pid)
    except Exception as ex:
        log(f"  {pid} failed on {suite}: {type(ex).__name__}: {ex}")
        return None
    emit("scored", predictor=pid, predictor_kind=kind, suite=suite, objective=r.get("objective"), specialist=base.name)
    r["fingerprint"] = fp
    if identity is not None:
        r["identity"] = identity
    p.save_result(pid, suite, r)
    return r


@stage("Score")
def run(
    name: str,
    experiments: list[str] | None = None,
    baselines: list[str] | None = None,
    suites: list[str] | None = None,
    force: bool = False,
    log: Callable[[str], None] = print,
) -> dict[str, dict[str, dict]]:
    p = Project(name)
    base = p.spec()
    exps = experiments if experiments is not None else p.experiments()
    bls = baselines if baselines is not None else base.baselines
    for suite in base.evals:
        if suites and suite.name not in suites:
            continue
        data = (*materialize(p, base, suite, log), suite.label_space(base.label_names))
        if not data[1]:
            log(f"  suite {suite.name}: empty, skipped")
            continue
        fp = fingerprint(p, base, suite)
        for job in _jobs(p, base, suite, exps, bls):
            if _cached_or_scored(p, base, suite.name, job, fp, force, data, log) is None:
                p.drop_result(job[1], suite.name)
                log(f"  {job[1]}: old {suite.name} result dropped from the board")
    return p.board()


def _board_row(pid: str, by_suite: dict[str, dict], suites: list[str], promoted: bool) -> dict[str, Any]:
    any_r = by_suite.get(suites[0]) or next(iter(by_suite.values()))
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
    row.update({f"{s}:n": by_suite.get(s, {}).get("n_test") for s in suites})
    row.update({f"{s}:more": summary(by_suite.get(s, {}).get("metrics") or {}) for s in suites})
    for s in suites:
        m = by_suite[s]["metrics"] if s in by_suite else None
        row[f"{s}:p/r"] = (m.get("precision"), m.get("recall")) if m is not None else None
    return row


def board_rows(name: str) -> tuple[list[str], list[dict]]:
    p = Project(name)
    base = p.spec()
    prim = base.primary_suite()
    suites = [prim, *(s.name for s in base.evals if s.name != prim)]
    rows = [_board_row(pid, by_suite, suites, pid == p.promoted) for pid, by_suite in p.board().items()]
    rows.sort(key=lambda r: -(r.get(prim) or -1))
    return suites, rows
