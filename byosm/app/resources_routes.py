from __future__ import annotations

import time
from functools import lru_cache
from pathlib import Path
from typing import Any

from fastapi import APIRouter

COSTS_TTL_S = 60


def resources_router(ws: Path) -> APIRouter:
    router = APIRouter()

    @router.get("/api/machine")
    def machine() -> dict[str, Any]:
        from ..machine import machine as probe

        return probe(ws)

    @router.get("/api/opportunities")
    def opportunities() -> list[dict[str, Any]]:
        from ..opportunities import trending

        return trending()

    @router.get("/api/costs")
    def costs() -> dict[str, Any]:
        return _costs(ws, int(time.time() // COSTS_TTL_S))

    return router


@lru_cache(maxsize=1)
def _costs(ws: Path, _bucket: int) -> dict[str, Any]:
    from ..costs import economics, settings

    return {"settings": settings(ws), "specialists": economics(ws)}
