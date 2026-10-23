from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel


class Kept(BaseModel):
    dataset: dict[str, Any]
    file: dict[str, Any]


class Url(BaseModel):
    url: str


def opendata_router(ws: Path) -> APIRouter:
    router = APIRouter()

    @router.get("/api/opendata/search")
    def search(q: str, source: str | None = None) -> list[dict[str, Any]]:
        from ..opendata.catalog import search as find

        return find(q, [source] if source else None)

    @router.get("/api/opendata/library")
    def library() -> list[dict[str, Any]]:
        from ..opendata.library import entries

        return entries(ws)

    @router.post("/api/opendata/library")
    def keep(body: Kept) -> list[dict[str, Any]]:
        from ..opendata.library import add

        return add(ws, body.dataset, body.file)

    @router.post("/api/opendata/library/remove")
    def drop(body: Url) -> list[dict[str, Any]]:
        from ..opendata.library import remove

        return remove(ws, body.url)

    @router.get("/api/opendata/preview")
    def preview(url: str, file: str | None = None) -> dict[str, Any]:
        from ..opendata.cache import cached, folder
        from ..opendata.preview import preview as look

        info = cached(url)
        if not info or not info["files"]:
            raise HTTPException(404, "not downloaded yet: download it first")
        name = file if file in info["files"] else info["files"][0]
        return {**look(folder(url) / name), "files": info["files"]}

    @router.get("/api/datasets/training")
    def training() -> list[dict[str, Any]]:
        from ..opendata.library import training as built

        return built(ws)

    return router
