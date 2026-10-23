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


class Source(BaseModel):
    url: str
    name: str = ""


class NewForecast(BaseModel):
    name: str
    spec: dict[str, Any]


class Request(BaseModel):
    request: str


def _table(url: str, file: str | None = None) -> Path:
    from ..opendata.cache import table

    try:
        return table(url, file)
    except KeyError as e:
        raise HTTPException(404, str(e).strip("'\"")) from e


def opendata_router(ws: Path) -> APIRouter:
    router = APIRouter()

    @router.get("/api/opendata/search")
    def search(q: str, source: str | None = None) -> list[dict[str, Any]]:
        from ..opendata.catalog import search as find
        from ..opendata.sources import portals

        return find(q, [source] if source else None, portals(ws))

    @router.get("/api/opendata/sources")
    def listed() -> list[dict[str, Any]]:
        from ..opendata.sources import listed as all_sources

        return all_sources(ws)

    @router.post("/api/opendata/sources")
    def add_source(body: Source) -> list[dict[str, Any]]:
        from ..opendata.sources import add

        return add(ws, body.url, body.name)

    @router.post("/api/opendata/sources/remove")
    def drop_source(body: Url) -> list[dict[str, Any]]:
        from ..opendata.sources import remove

        return remove(ws, body.url)

    @router.get("/api/opendata/requests")
    def requests() -> list[dict[str, Any]]:
        from ..opendata.finder import requests as asked

        return asked(ws)

    @router.post("/api/opendata/requests/remove")
    def drop_request(body: Request) -> list[dict[str, Any]]:
        from ..opendata.finder import forget

        return forget(ws, body.request)

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
        from ..opendata.cache import cached
        from ..opendata.preview import preview as look

        return {**look(_table(url, file)), "files": (cached(url) or {}).get("files", [])}

    @router.get("/api/opendata/join/suggest")
    def suggest(left: str, right: str) -> list[dict[str, Any]]:
        from ..opendata.join import suggest as keys

        return keys(_table(left), _table(right))

    @router.get("/api/opendata/join/preview")
    def joined(left: str, right: str, left_key: str, right_key: str) -> dict[str, Any]:
        from ..opendata.join import joined as join

        return join(_table(left), _table(right), left_key, right_key)

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

    @router.get("/api/datasets/training")
    def training() -> list[dict[str, Any]]:
        from ..opendata.library import training as built

        return built(ws)

    return router
