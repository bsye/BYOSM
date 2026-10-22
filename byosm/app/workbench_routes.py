from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel

from .errors import missing


class Models(BaseModel):
    models: list[str]


def _project(ws: Path, name: str) -> tuple[Any, Any]:
    from ..config import Project
    from ..spec import TaskSpec

    p = Project(name, ws)
    if not p.exists():
        raise KeyError(f"no specialist {name!r}")
    return p, TaskSpec.load(p.spec_path)


def _sets(ws: Path, name: str) -> list[dict[str, Any]]:
    from ..evals.testset import written_logs
    from ..evals.workbench import results

    p, spec = _project(ws, name)
    frozen = {e.name for e in spec.evals}
    return [
        {
            "suite": s,
            "frozen": s in frozen,
            "rows": [
                {"id": r["id"], "input": r["input"], "label": r["label"], "source": r.get("source")} for r in rows
            ],
            "results": results(p, s),
        }
        for s, rows in written_logs(p).items()
    ]


def _models(ws: Path, name: str) -> dict[str, Any]:
    from ..evals.workbench import OURS, chosen_models

    p, spec = _project(ws, name)
    return {"chosen": chosen_models(p, spec), "available": [OURS, *spec.baselines]}


def _choose(ws: Path, name: str, models: list[str]) -> dict[str, Any]:
    from ..evals.workbench import choose_models

    p, _ = _project(ws, name)
    choose_models(p, models)
    return _models(ws, name)


def workbench_router(ws: Path) -> APIRouter:
    router = APIRouter()

    @router.get("/api/specialists/{name}/written")
    def written(name: str) -> list:
        return missing(_sets, ws, name)

    @router.get("/api/specialists/{name}/test-models")
    def models(name: str) -> dict:
        return missing(_models, ws, name)

    @router.put("/api/specialists/{name}/test-models")
    def choose(name: str, body: Models) -> dict:
        return missing(_choose, ws, name, body.models)

    return router
