from __future__ import annotations

import json
import sqlite3
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ..app.store import JOB_FIELDS, JobStore
from ..auto.report import SUMMARY
from ..auto.state import AUTOPILOT
from ..data.store import read_file
from ..dataset_search import SUGGESTIONS
from ..evals.overfit import OVERFIT
from ..evals.workbench import MODELS, RESULTS_PREFIX
from ..health import LEAD
from ..hub.sections import CARD_NOTES
from ..improve.proposals import PROPOSALS
from ..project import Project
from ..spec import TaskSpec
from ..system.board import BOARD, RESULT
from ..system.bundle import GATE
from ..system.spec import SYSTEM_FILE, SystemProject, SystemSpec
from . import rows

JSON_REPORTS = {
    "overfit.json": OVERFIT,
    "proposals.json": PROPOSALS,
    "data-suggestions.json": SUGGESTIONS,
    "auto.json": AUTOPILOT,
    "test-models.json": MODELS,
    "cache/lead.json": LEAD,
}
TEXT_REPORTS = {"auto.md": SUMMARY, "CARD.md": CARD_NOTES}
STALE_EXPERIMENT_FILES = ("test_errors.json", "report.json")
WORKBENCH_RESULTS = ".results.json"
ROW_DIRS = ("specialists", "systems")
ROW_SKIP = {"export", "hub"}
OLD_JOB_DB = Path(".byosm") / "app.sqlite"
OLD_BOARD, OLD_GATE, TRACE_PREFIX = "board.json", "gate.json", "trace-"


def _json(f: Path) -> Any:
    return json.loads(f.read_text()) if f.exists() else None


def _experiment(p: Project, d: Path, done: list[Path]) -> None:
    files = {n: d / n for n in ("spec.yaml", "meta.json", "proposal.json", "dev_errors.json", "analysis.json")}
    meta = _json(files["meta.json"])
    if meta is not None and files["analysis.json"].exists():
        meta["refine_analyses"] = [_json(files["analysis.json"])]
    spec = TaskSpec.load(files["spec.yaml"]) if files["spec.yaml"].exists() else p.spec()
    p.save_experiment(
        d.name, spec=spec, meta=meta, proposal=_json(files["proposal.json"]), errors=_json(files["dev_errors.json"])
    )
    for r in sorted((d / "results").glob("*.json")):
        p.save_result(d.name, r.stem, _json(r))
    done += [f for f in files.values() if f.exists()] + [d / n for n in STALE_EXPERIMENT_FILES if (d / n).exists()]
    done += sorted((d / "results").glob("*.json"))


def _reports(p: Project, root: Path, done: list[Path]) -> None:
    for name, kind in JSON_REPORTS.items():
        if (root / name).exists():
            p.save_report(kind, _json(root / name))
            done.append(root / name)
    for name, kind in TEXT_REPORTS.items():
        if (root / name).exists():
            p.save_report(kind, (root / name).read_text())
            done.append(root / name)
    for f in sorted((root / "evals").glob(f"*{WORKBENCH_RESULTS}")):
        p.save_report(f"{RESULTS_PREFIX}{f.name.removesuffix(WORKBENCH_RESULTS)}", _json(f))
        done.append(f)


def import_specialist(root: Path, ws: Path) -> list[Path]:
    p = Project(root.name, ws)
    p.save_spec(TaskSpec.load(root / "spec.yaml"))
    done = [root / "spec.yaml"]
    for d in sorted((root / "experiments").glob("*")):
        if d.is_dir():
            _experiment(p, d, done)
    for f in sorted((root / "baselines").glob("*/*.json")):
        r = _json(f)
        p.save_result(r["predictor"], f.stem, r)
        done.append(f)
    board = _json(root / "board.json") or {}
    p.save_board(board)
    for pid, by_suite in p.board().items():
        for suite in by_suite:
            if suite not in board.get(pid, {}):
                p.drop_result(pid, suite)
    done.append(root / "board.json")
    if (root / "promoted").exists():
        p.promote((root / "promoted").read_text().strip())
        done.append(root / "promoted")
    _reports(p, root, done)
    return [f for f in done if f.exists()]


def import_rows(ws: Path, log: Callable[[str], None]) -> list[Path]:
    files = sorted(f for d in ROW_DIRS for f in (ws / d).glob(f"**/*{rows.SUFFIX}") if not ROW_SKIP & set(f.parts))
    for f in files:
        rows.write(f, read_file(f))
    log(f"  {len(files)} row files imported")
    return files


def import_jobs(ws: Path, log: Callable[[str], None]) -> list[Path]:
    old = ws / OLD_JOB_DB
    if not old.exists():
        return []
    db = sqlite3.connect(old)
    store = JobStore(ws)
    for row in db.execute(f"select {', '.join(JOB_FIELDS)} from jobs"):
        store.save({**dict(zip(JOB_FIELDS, row, strict=True)), "args": json.loads(row[1])})
    for job, line in db.execute("select job, line from lines order by rowid"):
        store.line(job, line)
    for job, data in db.execute("select job, data from events order by rowid"):
        store.event(job, json.loads(data))
    db.close()
    log(f"  {len(store.jobs())} jobs imported")
    return [old]


def import_system(root: Path, ws: Path) -> list[Path]:
    sp = SystemProject(root.name, ws)
    sp.save_spec(SystemSpec.load(root / SYSTEM_FILE))
    done = [root / SYSTEM_FILE]
    for name, kind in ((OLD_BOARD, BOARD), (OLD_GATE, GATE)):
        if (root / name).exists():
            sp.save_report(kind, _json(root / name))
            done.append(root / name)
    for f in sorted((root / "results").glob("*/*.json")):
        sp.save_report(f"{RESULT}:{f.parent.name}:{f.stem}", _json(f))
        done.append(f)
    for f in sorted((root / "evals").glob(f"*/{TRACE_PREFIX}*.json")):
        sp.save_report(f"trace:{f.parent.name}:{f.stem.removeprefix(TRACE_PREFIX)}", _json(f))
        done.append(f)
    return done


def import_workspace(ws: Path, log: Callable[[str], None]) -> list[Path]:
    done: list[Path] = []
    for spec in sorted((ws / "specialists").glob("*/spec.yaml")):
        files = import_specialist(spec.parent, ws)
        log(f"  {spec.parent.name}: {len(files)} files imported")
        done += files
    for spec in sorted((ws / "systems").glob(f"*/{SYSTEM_FILE}")):
        done += import_system(spec.parent, ws)
        log(f"  system {spec.parent.name} imported")
    return done + import_rows(ws, log) + import_jobs(ws, log)
