from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter


def techniques_router(ws: Path) -> APIRouter:
    router = APIRouter()

    @router.get("/api/techniques")
    def techniques() -> list[dict]:
        from ..improve.bandit import catalogue_board

        return catalogue_board(ws)

    return router
