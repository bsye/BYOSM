from __future__ import annotations

from pathlib import Path
from typing import Literal

from fastapi import APIRouter, HTTPException, UploadFile
from fastapi.responses import FileResponse

from ..data.check import DATA_CHECK
from ..data.request import DATA_REQUEST, REQUEST_DIR
from ..data.table import TABLE_SUFFIXES, TEMPLATE_STEM
from .catalog import project
from .errors import missing

UPLOAD_DIR = "uploads"
MAX_UPLOAD_BYTES = 50 * 1024 * 1024
SAFE_PUNCTUATION = "-_."


def _root(ws: Path, name: str) -> Path:
    return missing(project, ws, name).root


def safe_filename(raw: str) -> str:
    name = "".join(c if c.isalnum() or c in SAFE_PUNCTUATION else "_" for c in Path(raw).name)
    if name.startswith(".") or Path(name).suffix.lower() not in TABLE_SUFFIXES:
        raise HTTPException(422, f"upload a table: a file ending in {' or '.join(TABLE_SUFFIXES)}")
    return name


def _uploads(ws: Path, root: Path) -> list[str]:
    folder = root / UPLOAD_DIR
    found = sorted(folder.iterdir()) if folder.exists() else []
    return [str(f.relative_to(ws)) for f in found if f.suffix.lower() in TABLE_SUFFIXES]


def data_router(ws: Path) -> APIRouter:
    router = APIRouter(prefix="/api/specialists/{name}")

    @router.get("/request")
    def request(name: str) -> dict:
        root = _root(ws, name)
        req = root / REQUEST_DIR
        return {
            "request": missing(project, ws, name).report(DATA_REQUEST),
            "templates": [s[1:] for s in TABLE_SUFFIXES if (req / f"{TEMPLATE_STEM}{s}").exists()],
            "uploads": _uploads(ws, root),
        }

    @router.get("/request/template.{kind}")
    def template(name: str, kind: Literal["csv", "xlsx"]) -> FileResponse:
        path = _root(ws, name) / REQUEST_DIR / f"{TEMPLATE_STEM}.{kind}"
        if not path.exists():
            raise HTTPException(404, f"{name} has no data request yet: run request-data first")
        return FileResponse(path, filename=f"{name}-{path.name}")

    @router.post("/uploads")
    async def upload(name: str, file: UploadFile) -> dict:
        folder = _root(ws, name) / UPLOAD_DIR
        dest = folder / safe_filename(file.filename or "")
        body = await file.read(MAX_UPLOAD_BYTES + 1)
        if len(body) > MAX_UPLOAD_BYTES:
            raise HTTPException(413, f"the table is larger than {MAX_UPLOAD_BYTES // 1024 // 1024} MB")
        folder.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(body)
        return {"path": str(dest.relative_to(ws)), "bytes": len(body)}

    @router.get("/check")
    def check(name: str) -> dict:
        return {"report": missing(project, ws, name).report(DATA_CHECK)}

    return router
