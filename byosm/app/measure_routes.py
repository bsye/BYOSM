from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel

from .errors import missing
from .jobs import Runner


class MeasureRequest(BaseModel):
    names: list[str] | None = None


def measure_router(ws: Path, runner: Runner) -> APIRouter:
    from ..measure.conduct import Conductor
    from ..measure.run import board

    router = APIRouter()
    conductor = Conductor(ws, runner, runner.hub.publish)
    router.on_shutdown.append(conductor.stop)

    def shown() -> dict[str, Any]:
        return board(ws, {j.id: j for j in runner.snapshot()}, conductor.error)

    @router.get("/api/measure")
    def measure_plan() -> dict[str, Any]:
        return shown()

    @router.post("/api/measure")
    def measure(body: MeasureRequest | None = None) -> dict[str, Any]:
        missing(conductor.queue, (body or MeasureRequest()).names)
        return shown()

    return router
