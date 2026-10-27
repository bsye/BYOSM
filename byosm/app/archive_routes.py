from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter
from pydantic import BaseModel


class Archive(BaseModel):
    archived: bool


def archive_router(ws: Path) -> APIRouter:
    router = APIRouter()

    @router.post("/api/specialists/{name}/archive")
    def archive(name: str, body: Archive) -> dict:
        from ..archive import set_archived

        return {"name": name, "archived": name in set_archived(ws, name, body.archived)}

    return router
