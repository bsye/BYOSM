from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, UploadFile
from pydantic import BaseModel

from .occlusion import MAX_BYTES


class PageRequest(BaseModel):
    file: str
    label: str | None = None


def page_router(ws: Path) -> APIRouter:
    router = APIRouter(prefix="/api/specialists/{name}")

    def _model(name: str) -> Any:
        from .bundles import bundle

        try:
            model = bundle(ws, name)
        except FileNotFoundError as e:
            raise HTTPException(404, str(e)) from e
        if model.m["modality"] != "document" or model.m["task_type"] != "classify":
            raise HTTPException(422, "the page view works for document classifiers")
        return model

    def _path(file: str) -> Path:
        from .occlusion import tried_path

        try:
            return tried_path(file)
        except ValueError as e:
            raise HTTPException(422, str(e)) from e
        except FileNotFoundError as e:
            raise HTTPException(404, str(e)) from e

    @router.post("/tried")
    async def tried(name: str, file: UploadFile) -> dict[str, str]:
        from .occlusion import save_tried

        _model(name)
        try:
            return {"file": save_tried(await file.read(MAX_BYTES + 1), file.filename or "")}
        except ValueError as e:
            raise HTTPException(422, str(e)) from e

    @router.post("/page")
    def page(name: str, body: PageRequest) -> dict[str, Any]:
        from .occlusion import page as read

        return read(_model(name), _path(body.file))

    @router.post("/page/{row}")
    def row(name: str, row: int, body: PageRequest) -> dict[str, Any]:
        from .occlusion import occlusion_row

        model = _model(name)
        if body.label not in model.labels:
            raise HTTPException(422, f"label is one of {', '.join(model.labels)}")
        try:
            return {"row": row, "drops": occlusion_row(model, _path(body.file), row, body.label)}
        except ValueError as e:
            raise HTTPException(422, str(e)) from e

    return router
