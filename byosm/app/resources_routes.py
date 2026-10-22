from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import APIRouter


def resources_router(ws: Path) -> APIRouter:
    router = APIRouter()

    @router.get("/api/machine")
    def machine() -> dict[str, Any]:
        from ..machine import machine as probe

        return probe(ws)

    return router
