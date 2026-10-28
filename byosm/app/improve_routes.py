from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import APIRouter

from .lab_routes import lab_router
from .scout_routes import scout_router
from .techniques_routes import techniques_router


def improve_router(ws: Path, runner: Any) -> APIRouter:
    router = APIRouter()
    for r in (techniques_router(ws), lab_router(ws), scout_router(ws, runner)):
        router.include_router(r)
    return router
