from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from sqlalchemy import select

from ..db import get_or_add, session
from ..db.models import WorkspaceReport
from .cache import cached

LIBRARY = "open-datasets"


def entries(ws: Path) -> list[dict[str, Any]]:
    with session(ws) as s:
        data = s.scalar(select(WorkspaceReport.data).where(WorkspaceReport.kind == LIBRARY))
    return [{**e, "cached": cached(e["file"]["url"])} for e in (data or [])]


def _save(ws: Path, rows: list[dict[str, Any]]) -> None:
    with session(ws) as s:
        get_or_add(s, WorkspaceReport, kind=LIBRARY).data = rows


def add(ws: Path, dataset: dict[str, Any], file: dict[str, Any]) -> list[dict[str, Any]]:
    keep = [e for e in entries(ws) if e["file"]["url"] != file["url"]]
    entry = {k: dataset.get(k) for k in ("source", "id", "title", "publisher", "licence", "url", "modified")}
    _save(
        ws,
        [{k: v for k, v in e.items() if k != "cached"} for e in keep] + [{**entry, "file": file, "added": time.time()}],
    )
    return entries(ws)


def remove(ws: Path, url: str) -> list[dict[str, Any]]:
    _save(ws, [{k: v for k, v in e.items() if k != "cached"} for e in entries(ws) if e["file"]["url"] != url])
    return entries(ws)


def training(ws: Path) -> list[dict[str, Any]]:
    from ..data.store import Dataset
    from ..project import Project, list_projects

    out = []
    for name in list_projects(ws):
        p = Project(name, ws)
        spec = p.spec()
        data = Dataset(p.data_dir_for(spec))
        if not data.built():
            continue
        summary = data.summary()
        out.append(
            {
                "name": name,
                "title": spec.title or name,
                "modality": spec.modality,
                "labels": len(spec.labels),
                "splits": {s: v["n"] for s, v in summary.items()},
                "sources": sorted({s.dataset for s in spec.data.hf_sources}),
            }
        )
    return out
