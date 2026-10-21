from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from ..auto.standing import best_baseline
from ..config import Project, list_projects
from ..data.store import SPLITS, Dataset
from ..spec import TaskSpec

MAX_PAGE = 500
RESULT_FIELDS = ("objective", "calib_objective", "latency_ms", "params", "n_test", "n_calib")


def _project(ws: Path, name: str) -> Project:
    p = Project(name, ws)
    if not p.exists():
        raise KeyError(f"no specialist {name!r}")
    return p


def _result(r: dict) -> dict[str, Any]:
    return {**{k: r.get(k) for k in RESULT_FIELDS}, "ci": r.get("metrics", {}).get("ci95")}


def summary(ws: Path, name: str) -> dict[str, Any]:
    from ..export.bundle import is_exported

    p = _project(ws, name)
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
        "promoted": p.promoted,
        "score": best.get("objective"),
        "rival": {"name": rival[0], "score": board[rival[0]][prim].get("objective")} if rival else None,
        "experiments": p.experiments(),
        "exported": is_exported(p.root / "export"),
    }


def specialists(ws: Path) -> list[dict[str, Any]]:
    return [summary(ws, n) for n in list_projects(ws)]


def detail(ws: Path, name: str) -> dict[str, Any]:
    p = _project(ws, name)
    board = {pid: {suite: _result(r) for suite, r in by.items()} for pid, by in p.board().items()}
    return {
        **summary(ws, name),
        "spec": TaskSpec.load(p.spec_path).model_dump(mode="json"),
        "board": board,
        "baselines": TaskSpec.load(p.spec_path).baselines,
    }


def save_spec(ws: Path, name: str, body: dict) -> dict[str, Any]:
    p = _project(ws, name)
    spec = TaskSpec(**{**body, "name": name})
    spec.save(p.spec_path)
    return spec.model_dump(mode="json")


def delete(ws: Path, name: str) -> None:
    shutil.rmtree(_project(ws, name).root)


def rows(ws: Path, name: str, split: str, offset: int, limit: int) -> dict[str, Any]:
    if split not in SPLITS:
        raise KeyError(f"split must be one of {', '.join(SPLITS)}")
    p = _project(ws, name)
    data = Dataset(p.data_dir_for(TaskSpec.load(p.spec_path))).load(split)
    return {"total": len(data), "rows": data[offset : offset + min(limit, MAX_PAGE)]}


def systems(ws: Path) -> list[dict[str, Any]]:
    from ..system import SystemProject

    root = ws / "systems"
    names = sorted(d.name for d in root.glob("*") if (d / "system.yaml").exists()) if root.exists() else []
    return [system(ws, n) for n in names if SystemProject(n, ws).spec_path.exists()]


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
