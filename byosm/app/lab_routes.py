from __future__ import annotations

from pathlib import Path
from typing import Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel


class Decision(BaseModel):
    status: Literal["approved", "rejected"]


def lab_router(ws: Path) -> APIRouter:
    from ..project import Project

    router = APIRouter()

    def _project(name: str) -> Project:
        p = Project(name, ws)
        if not p.exists():
            raise HTTPException(404, f"no specialist {name!r}")
        return p

    @router.get("/api/lab/proven")
    def proven() -> list[dict]:
        from ..lab.proven import registry

        return sorted(registry(ws).values(), key=lambda e: (-len(e["won_on"]), e["name"]))

    @router.get("/api/specialists/{name}/lab")
    def methods(name: str) -> list[dict]:
        from ..lab.drafts import drafts

        return drafts(_project(name))

    @router.post("/api/specialists/{name}/lab/{method}")
    def decide(name: str, method: str, body: Decision) -> dict:
        from ..lab.drafts import decide as decided

        try:
            return decided(_project(name), method, body.status)
        except KeyError as e:
            raise HTTPException(404, str(e)) from e
        except ValueError as e:
            raise HTTPException(422, str(e)) from e

    return router
