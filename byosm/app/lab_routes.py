from __future__ import annotations

from pathlib import Path
from typing import Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel


class Decision(BaseModel):
    status: Literal["approved", "rejected"]


def lab_router(ws: Path) -> APIRouter:
    from ..lab.owners import Owner, owner

    router = APIRouter()

    def _owner(name: str) -> Owner:
        try:
            return owner(name, ws)
        except KeyError as e:
            raise HTTPException(404, str(e).strip("'\"")) from e

    @router.get("/api/lab/proven")
    def proven() -> list[dict]:
        from ..lab.proven import registry

        return sorted(registry(ws).values(), key=lambda e: (-len(e["won_on"]), e["name"]))

    @router.get("/api/specialists/{name}/lab")
    def methods(name: str) -> list[dict]:
        from ..lab.drafts import drafts

        return drafts(_owner(name))

    @router.post("/api/specialists/{name}/lab/{method}")
    def decide(name: str, method: str, body: Decision) -> dict:
        from ..lab.drafts import decide as decided

        try:
            return decided(_owner(name), method, body.status)
        except KeyError as e:
            raise HTTPException(404, str(e)) from e
        except ValueError as e:
            raise HTTPException(422, str(e)) from e

    return router
