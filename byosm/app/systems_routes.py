from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

MAX_TYPED_CHARS = 300


class Typed(BaseModel):
    text: str


def systems_router(ws: Path, auth: Any) -> APIRouter:
    from ..rt import System, is_system

    router = APIRouter(dependencies=[auth])
    loaded: dict[str, System] = {}

    @router.post("/api/systems/{name}/type")
    def typed(name: str, body: Typed) -> dict:
        bundle = ws / "systems" / name / "export"
        if not is_system(bundle):
            raise HTTPException(404, f"{name} is not exported: run the system build job first")
        system = loaded.get(name) or loaded.setdefault(name, System(bundle))
        text = body.text[:MAX_TYPED_CHARS]
        try:
            stream = system.stream()
        except ValueError as e:
            raise HTTPException(422, str(e)) from e
        trace, fired_at = [], None
        for i in range(1, len(text) + 1):
            step = stream.step(text[:i])
            if step["fired"] and fired_at is None:
                fired_at = i
            trace.append({"chars": i, "action": step["action"], "confidence": step["confidence"]})
        return {"trace": trace, "fired_at": fired_at, "frame": stream.finish(text) if text.strip() else None}

    return router
