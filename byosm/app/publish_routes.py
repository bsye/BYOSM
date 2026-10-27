from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel

from ..config import PUBLISH_LICENSE


class Target(BaseModel):
    repo: str
    license: str = PUBLISH_LICENSE


class CardEdit(BaseModel):
    card: str
    instruction: str


class Card(BaseModel):
    repo: str
    card: str


class Trial(BaseModel):
    text: str | None = None


def publish_router(ws: Path) -> APIRouter:
    router = APIRouter()

    @router.post("/api/specialists/{name}/publish/preview")
    def preview(name: str, body: Target) -> dict[str, Any]:
        from ..hub.preview import preview

        return preview(name, body.repo, body.license)

    @router.post("/api/specialists/{name}/publish/edit")
    def edit(name: str, body: CardEdit) -> dict[str, Any]:
        from ..hub.preview import edit

        return edit(body.card, body.instruction)

    @router.post("/api/specialists/{name}/publish/card")
    def save(name: str, body: Card) -> dict[str, str]:
        from ..hub.preview import save_card

        return {"path": str(save_card(name, body.repo, body.card))}

    @router.post("/api/specialists/{name}/publish/test")
    def test(name: str, body: Trial) -> dict[str, Any]:
        from ..hub.preview import test_bundle

        return test_bundle(name, body.text)

    return router
