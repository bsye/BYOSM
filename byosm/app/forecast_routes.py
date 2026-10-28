from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel


class History(BaseModel):
    history: list[float | None]
    freq: str
    config: str | None = None
    start: str | None = None


class Entity(BaseModel):
    key: str | None = None
    row: dict[str, Any] | None = None
    usual: list[str] = []


def forecast_router(ws: Path) -> APIRouter:
    router = APIRouter()

    @router.get("/api/forecasts")
    def forecasts() -> list[dict[str, Any]]:
        from ..tabular.task import forecasts as saved

        return saved(ws)

    @router.delete("/api/forecasts/{name}")
    def drop_forecast(name: str, confirm: str) -> list[dict[str, Any]]:
        from ..tabular.task import forecasts, remove

        if confirm != name:
            raise HTTPException(422, "pass confirm=<name> to delete a forecast")
        try:
            remove(ws, name)
        except KeyError as e:
            raise HTTPException(404, str(e).strip("'\"")) from e
        return forecasts(ws)

    @router.get("/api/forecasts/{name}/scored.csv")
    def scored(name: str) -> FileResponse:
        from ..tabular.task import scored_path

        path = scored_path(ws, name)
        if not path.exists():
            raise HTTPException(404, "run the forecast first")
        return FileResponse(path, filename=f"{name}-scored.csv")

    @router.get("/api/series")
    def series() -> list[dict[str, Any]]:
        from ..series.board import reports

        return reports(ws)

    @router.get("/api/series/{domain}")
    def forecaster(domain: str) -> dict[str, Any]:
        from ..series.board import report_of
        from ..series.specialist import summary

        found = report_of(ws, domain)
        if found is None:
            raise HTTPException(404, f"no forecasting domain named {domain}")
        from ..series.board import promoted_board
        from ..series.health import health
        from ..series.saved import live_version
        from ..series.steps import next_steps

        live = live_version(ws, domain, found)
        board = {**promoted_board(found), **summary(found)}
        return {**found, **board, "health": health(found), "next": next_steps(ws, found), "live": live}

    @router.get("/api/series/{domain}/example")
    def example(domain: str, config: str, index: int = 0) -> dict[str, Any]:
        from ..series.saved import example

        try:
            return example(domain, config, index)
        except KeyError as e:
            raise HTTPException(404, str(e).strip("'\"")) from e

    @router.post("/api/series/{domain}/forecast")
    def forecast(domain: str, body: History) -> dict[str, Any]:
        from ..series.board import report_of
        from ..series.saved import forecast, live_version

        found = report_of(ws, domain) or {}
        version = live_version(ws, domain, found)
        if version is None:
            raise HTTPException(409, "train a version first: Keep training saves the model it builds")
        made = forecast(ws, domain, version, body.history, body.freq, body.config, body.start)
        return {"version": version, **made}

    @router.get("/api/forecasts/{name}/inputs")
    def inputs(name: str) -> dict[str, Any]:
        from ..tabular.predict import inputs as needed

        return needed(ws, name)

    @router.post("/api/forecasts/{name}/predict")
    def predict(name: str, body: Entity) -> dict[str, Any]:
        from ..tabular.predict import lookup, predict

        row = lookup(ws, name, body.key, lambda m: None) if body.key else body.row
        if row is None:
            raise HTTPException(404, f"no row with {body.key}")
        try:
            return predict(ws, name, row, lambda m: None, body.usual)
        except KeyError as e:
            raise HTTPException(404, str(e).strip("'\"")) from e
        except ValueError as e:
            raise HTTPException(422, str(e)) from e

    return router
