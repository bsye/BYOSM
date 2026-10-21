from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from ..auto.standing import best_baseline
from ..config import Project, list_projects
from ..data.store import SPLITS, Dataset
from ..hub.facts import suite_dataset
from ..improve.proposals import PROPOSALS_FILE
from ..spec import TaskSpec

MAX_PAGE = 500
RESULT_FIELDS = ("objective", "calib_objective", "latency_ms", "params", "n_test", "n_calib")
SUITE_FIELDS = ("name", "kind", "primary", "written_by", "judged_by")


def project(ws: Path, name: str) -> Project:
    p = Project(name, ws)
    if not p.exists():
        raise KeyError(f"no specialist {name!r}")
    return p


def _result(r: dict) -> dict[str, Any]:
    return {**{k: r.get(k) for k in RESULT_FIELDS}, "ci": r.get("metrics", {}).get("ci95")}


def summary(ws: Path, name: str) -> dict[str, Any]:
    from ..export.bundle import is_exported

    p = project(ws, name)
    s = TaskSpec.load(p.spec_path)
    prim = s.primary_suite()
    board = p.board()
    best = board.get(p.promoted or "", {}).get(prim, {})
    rival = best_baseline(board, prim)
    return {
        "name": name,
        "description": s.description,
        "modality": s.modality,
        "task_type": s.task_type,
        "labels": len(s.labels),
        "metric": s.objective.metric,
        "primary_suite": prim,
        "benchmark": suite_dataset(s, prim)[0].rpartition("/")[2],
        "promoted": p.promoted,
        "score": best.get("objective"),
        "rival": {"name": rival[0], "score": board[rival[0]][prim].get("objective")} if rival else None,
        "experiments": p.experiments(),
        "exported": is_exported(p.root / "export"),
    }


def specialists(ws: Path) -> list[dict[str, Any]]:
    from .teams import members, team_names, team_summary

    part_of = members(ws)
    singles = [{**summary(ws, n), "kind": "model", "part_of": part_of.get(n)} for n in list_projects(ws)]
    return singles + [team_summary(ws, t) for t in team_names(ws)]


def _parent(p: Project, exp: str) -> str | None:
    f = p.exp_dir(exp) / "proposal.json"
    return json.loads(f.read_text()).get("parent") if f.exists() else None


def _run(p: Project, exp: str) -> dict[str, Any]:
    m = p.meta(exp)
    return {
        "name": exp,
        "parent": _parent(p, exp),
        "params": m.get("params"),
        "n_train": m.get("n_train"),
        "train_seconds": m.get("train_seconds"),
        "best_epoch": m.get("best_epoch"),
        "history": [
            {"epoch": h["epoch"] + 1, "loss": h["loss"], "dev": h["dev_objective"]} for h in m.get("history", [])
        ],
    }


def detail(ws: Path, name: str) -> dict[str, Any]:
    p = project(ws, name)
    spec = TaskSpec.load(p.spec_path)
    board = {pid: {suite: _result(r) for suite, r in by.items()} for pid, by in p.board().items()}
    dd = p.data_dir_for(spec)
    return {
        **summary(ws, name),
        "spec": spec.model_dump(mode="json"),
        "board": board,
        "suites": [{k: getattr(e, k) for k in SUITE_FIELDS} for e in spec.evals],
        "baselines": spec.baselines,
        "runs": [_run(p, e) for e in p.experiments()],
        "proposals": json.loads(proposals.read_text()) if (proposals := p.root / PROPOSALS_FILE).exists() else None,
        "data": Dataset(dd).summary() if (dd / "train.jsonl").exists() else None,
        "overfit": json.loads(p.overfit_path.read_text()) if p.overfit_path.exists() else None,
    }


def save_spec(ws: Path, name: str, body: dict) -> dict[str, Any]:
    p = project(ws, name)
    spec = TaskSpec(**{**body, "name": name})
    spec.save(p.spec_path)
    return spec.model_dump(mode="json")


def delete(ws: Path, name: str) -> None:
    shutil.rmtree(project(ws, name).root)


def rows(ws: Path, name: str, split: str, offset: int, limit: int) -> dict[str, Any]:
    if split not in SPLITS:
        raise KeyError(f"split must be one of {', '.join(SPLITS)}")
    p = project(ws, name)
    data = Dataset(p.data_dir_for(TaskSpec.load(p.spec_path))).load(split)
    return {"total": len(data), "rows": data[offset : offset + min(limit, MAX_PAGE)]}


def systems(ws: Path) -> list[dict[str, Any]]:
    from ..system import SystemProject
    from .teams import team_names

    return [system(ws, n) for n in team_names(ws) if SystemProject(n, ws).spec_path.exists()]


def system(ws: Path, name: str) -> dict[str, Any]:
    from ..system import SystemProject

    sp = SystemProject(name, ws)
    if not sp.spec_path.exists():
        raise KeyError(f"no system {name!r}")
    board = sp.root / "board.json"
    return {
        "name": name,
        "spec": sp.spec().model_dump(mode="json"),
        "board": json.loads(board.read_text()) if board.exists() else None,
        "exported": (sp.export_dir / "system.json").exists(),
    }
