from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel


class NewForecast(BaseModel):
    name: str
    spec: dict[str, Any]


class Entity(BaseModel):
    key: str | None = None
    row: dict[str, Any] | None = None


def forecast_router(ws: Path) -> APIRouter:
    router = APIRouter()

    @router.get("/api/forecasts")
    def forecasts() -> list[dict[str, Any]]:
        from ..tabular.task import forecasts as saved

        return saved(ws)

    @router.post("/api/forecasts")
    def new_forecast(body: NewForecast) -> list[dict[str, Any]]:
        from pydantic import ValidationError

        from ..tabular.task import create, forecasts

        try:
            create(ws, body.name, body.spec)
        except (ValueError, ValidationError) as e:
            raise HTTPException(422, str(e)) from e
        return forecasts(ws)

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
        from ..series.run import reports

        return reports(ws)

    @router.get("/api/series/{domain}")
    def forecaster(domain: str) -> dict[str, Any]:
        from ..series.run import reports
        from ..series.specialist import summary

        found = next((r for r in reports(ws) if r["domain"] == domain), None)
        if found is None:
            raise HTTPException(404, f"no forecasting domain named {domain}")
        return {**found, **summary(found)}

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
            return predict(ws, name, row, lambda m: None)
        except KeyError as e:
            raise HTTPException(404, str(e).strip("'\"")) from e

    return router
