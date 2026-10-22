from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel

from ..data.panel import load_panel
from ..data.review import load_review, reviewed_labels, save_review
from .catalog import project
from .errors import missing


class Review(BaseModel):
    input: str
    label: str


def panel(ws: Path, name: str) -> dict[str, Any]:
    p = project(ws, name)
    spec = p.spec()
    dd = p.data_dir_for(spec)
    human = reviewed_labels(p.reviewed_path)
    teachers = spec.data.teachers
    rows = [
        {
            "input": r["input"],
            "panel": r["panel"],
            "sure": r["sure"],
            "votes": {t: r.get(t) for t in teachers},
            "reviewed": human.get(str(r["input"])),
        }
        for r in load_review(dd)
    ]
    return {
        "teachers": teachers,
        "labels": spec.label_names,
        "summary": load_panel(dd),
        "n_reviewed": len(human),
        "review": rows,
    }


def review(ws: Path, name: str, body: Review) -> dict[str, Any]:
    p = project(ws, name)
    if body.label not in p.spec().label_index:
        raise ValueError(f"{body.label!r} is not one of {name}'s labels")
    return save_review(p.reviewed_path, body.input, body.label)


def router(ws: Path) -> APIRouter:
    routes = APIRouter(prefix="/api/specialists/{name}/panel")

    @routes.get("")
    def get_panel(name: str) -> dict:
        return missing(panel, ws, name)

    @routes.put("/review")
    def put_review(name: str, body: Review) -> dict:
        return missing(review, ws, name, body)

    return routes
