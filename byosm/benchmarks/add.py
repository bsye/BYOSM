from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from ..config import STATE_DIR
from ..data.hf_source.cards import licence
from ..data.store import write_file
from ..format import fmt
from ..project import Project
from ..spec import EvalSuite
from . import protocol
from .catalog import BENCHMARKS, Benchmark, Log, fitting

FOLDER = "benchmarks"
LOCAL = "local"


def table_path(ws: Path, name: str, suite: str) -> Path:
    return ws / STATE_DIR / FOLDER / name / f"{suite}.jsonl"


def chosen(p: Project, names: list[str]) -> list[Benchmark]:
    spec = p.spec()
    fit = sorted(b.name for b in fitting(spec))
    if unknown := [n for n in names if n not in BENCHMARKS]:
        raise ValueError(f"no public benchmark {', '.join(unknown)}; known: {', '.join(BENCHMARKS)}")
    if unfit := [n for n in names if n not in fit]:
        raise ValueError(
            f"{', '.join(unfit)} does not fit {p.name}'s labels or modality; fitting: {', '.join(fit) or 'none'}"
        )
    present = {e.name for e in spec.evals}
    return [BENCHMARKS[n] for n in names or [n for n in fit if n not in present]]


def _counts(rows: list[dict]) -> str:
    found = Counter((r["split"], r["label"]) for r in rows)
    return ", ".join(f"{split} {label} {n}" for (split, label), n in sorted(found.items()))


def _discard(p: Project, suite: EvalSuite) -> None:
    from ..evals.suites import freeze, suite_dir

    freeze(suite_dir(p, suite), suite, [], [])


def add(p: Project, b: Benchmark, log: Log) -> dict[str, Any]:
    spec = p.spec()
    if any(e.name == b.name for e in spec.evals):
        raise ValueError(f"suite {b.name} exists on {p.name}")
    rows = b.rows(spec, log)
    path = table_path(p.ws, p.name, b.name)
    write_file(path, rows)
    log(f"  {b.name}: {len(rows)} rows from {b.dataset} ({_counts(rows)}) in {path}")
    suite = EvalSuite(name=b.name, kind=LOCAL, path=str(path), dataset=b.dataset, licence=licence(b.dataset))
    check = protocol.run(p, spec, suite, b.kind, rows, log)
    for c in check["checks"]:
        verdict = "ok" if c["ok"] else "MISMATCH"
        log(f"  protocol check, {c['metric']}: expected {fmt(c['expected'])}, scored {fmt(c['scored'])}, {verdict}")
    for model, found in (check.get("published") or {}).items():
        log(f"  published on this split, {model}: " + ", ".join(f"{k} {fmt(v)}" for k, v in found.items()))
    if not check["passed"]:
        _discard(p, suite)
        raise ValueError(f"{b.name} failed its protocol check on {p.name}: the suite was not added")
    spec.evals.append(suite)
    p.save_spec(spec)
    return check
