from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from . import data as data_stage
from . import evals
from .config import Project, load_config
from .data import synth_text
from .data.review import reviewed_labels
from .data.store import Dataset
from .events import stage
from .spec import TaskSpec
from .train import fit


def next_exp(p: Project) -> str:
    n = 1
    while p.exp_dir(f"v{n}").exists():
        n += 1
    return f"v{n}"


def exp_spec(p: Project, exp: str) -> TaskSpec:
    f = p.exp_dir(exp) / "spec.yaml"
    return TaskSpec.load(f if f.exists() else p.spec_path)


@stage("Prepare data")
def ensure_data(
    p: Project,
    spec: TaskSpec,
    cfg: dict,
    log: Callable[[str], None],
    force: bool = False,
    seconds: float | None = None,
) -> Path:
    dd = p.data_dir_for(spec)
    if force or not (dd / "train.jsonl").exists():
        log(f"[data] {spec.name} -> data/{spec.data_key()}")
        dd.mkdir(parents=True, exist_ok=True)
        summary = data_stage.build(spec, dd, cfg, log, reviewed_labels(p.reviewed_path), seconds)
        for s, v in summary.items():
            log(f"  {s}: {v['n']} examples {v['labels'] if spec.task_type == 'classify' else ''}")
    else:
        log(f"[data] reusing data/{spec.data_key()} ({len(Dataset(dd).load('train'))} train)")
    return dd


def refine(p: Project, spec: TaskSpec, cfg: dict, exp_dir: Path, data_dir: Path, log: Callable[[str], None]) -> int:
    errors = json.loads((exp_dir / "dev_errors.json").read_text())
    if not errors:
        return 0
    log(f"[refine] analyzing {len(errors)} dev errors")
    jobs, analysis = synth_text.refine_jobs(spec, errors, spec.train.refine_examples, cfg)
    (exp_dir / "analysis.json").write_text(json.dumps(analysis, indent=2, ensure_ascii=False))
    for pat in analysis.get("patterns", []):
        log(f"  pattern: {pat['name']} - {pat['explanation'][:140]}")
    ds = Dataset(data_dir)
    prior = ds.load("train") + ds.load("dev") + ds.load("test")
    rows = synth_text.generate(spec, 0, "train", cfg, against=prior, jobs=jobs, log=log)
    ds.add("train", rows)
    log(f"  added {len(rows)} targeted training examples")
    return len(rows)


def build(
    name: str,
    exp: str | None = None,
    log: Callable[[str], None] = print,
    force_data: bool = False,
    data_seconds: float | None = None,
) -> dict[str, Any]:
    p = Project(name).ensure()
    cfg = load_config(p.ws)
    exp = exp or next_exp(p)
    spec = exp_spec(p, exp)
    ed = p.exp_dir(exp)
    ed.mkdir(parents=True, exist_ok=True)
    log(f"[experiment] {name}/{exp}")

    dd = ensure_data(p, spec, cfg, log, force_data, data_seconds)
    can_refine = spec.modality == "text" and spec.data.synthetic
    rounds = []
    for r in range(spec.train.refine_rounds + 1):
        log(f"[train] round {r}")
        meta = fit(spec, dd, ed, log)
        rounds.append({"round": r, "dev_objective": meta["dev"]["objective"], "n_train": meta["n_train"]})
        if meta["dev"]["objective"] >= spec.objective.target or not can_refine or r == spec.train.refine_rounds:
            break
        if refine(p, spec, cfg, ed, dd, log) == 0:
            break
    meta = json.loads((ed / "meta.json").read_text())
    meta["rounds"], meta["data_key"] = rounds, spec.data_key()
    (ed / "meta.json").write_text(json.dumps(meta, indent=2))

    log("[eval] scoring on every suite (+ baselines, cached)")
    evals.run(name, log=log)
    return maybe_promote(p, exp, log)


def maybe_promote(p: Project, exp: str, log: Callable[[str], None] = print) -> dict[str, Any]:
    base = TaskSpec.load(p.spec_path)
    prim = base.primary_suite()
    board = p.board()
    a, b = board.get(exp, {}).get(prim), board.get(p.promoted or "", {}).get(prim) if p.promoted else None
    mine, cur = evals.selection_value(a), evals.selection_value(b)
    mine = -1 if mine is None else mine
    cur = -1 if cur is None else cur
    cmp = evals.compare(base, a, b) if b else None
    sig = f", P(better)={cmp['p_better']:.2f}" if cmp else ""
    test = f"; test {a['objective']:.4f}" if a else ""
    if mine > cur:
        p.promote(exp)
        log(f"  promoted {exp}: {prim} calib {base.objective.metric} {mine:.4f} (was {cur:.4f}{sig}){test}")
    else:
        log(f"  kept {p.promoted}: {prim} calib {cur:.4f} >= {exp} {mine:.4f}{sig}{test}")
    return board.get(exp, {})
