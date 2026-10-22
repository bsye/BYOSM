from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import APIRouter, Response
from pydantic import BaseModel

from .errors import missing


class Models(BaseModel):
    models: list[str]


class LlmRival(BaseModel):
    config: dict[str, Any]


def _project(ws: Path, name: str) -> tuple[Any, Any]:
    from ..project import Project

    p = Project(name, ws)
    if not p.exists():
        raise KeyError(f"no specialist {name!r}")
    return p, p.spec()


def _sets(ws: Path, name: str) -> list[dict[str, Any]]:
    from ..evals.test_sets import summaries

    return summaries(*_project(ws, name))


def _page(ws: Path, name: str, suite: str, offset: int, limit: int) -> dict[str, Any]:
    from ..evals.test_sets import page

    return page(*_project(ws, name), suite, offset, limit)


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

    @router.get("/api/specialists/{name}/written/{suite}/rows")
    def rows(name: str, suite: str, offset: int = 0, limit: int = 100) -> dict:
        return missing(_page, ws, name, suite, offset, limit)

    @router.get("/api/specialists/{name}/written/{suite}.xlsx")
    def export(name: str, suite: str) -> Response:
        from ..evals.workbench_export import XLSX_TYPE, results_workbook

        p, spec = missing(_project, ws, name)
        body = missing(results_workbook, p, spec, suite)
        filename = f"{name}-{suite}-results.xlsx"
        return Response(
            body, media_type=XLSX_TYPE, headers={"Content-Disposition": f'attachment; filename="{filename}"'}
        )

    @router.post("/api/specialists/{name}/llm-rivals")
    def add_rival(name: str, body: LlmRival) -> dict:
        from ..llm_rivals import add_llm_rival

        p, _ = missing(_project, ws, name)
        return {"rival": missing(add_llm_rival, p, body.config)}

    @router.get("/api/specialists/{name}/test-models")
    def models(name: str) -> dict:
        return missing(_models, ws, name)

    @router.put("/api/specialists/{name}/test-models")
    def choose(name: str, body: Models) -> dict:
        return missing(_choose, ws, name, body.models)

    return router
