from __future__ import annotations

from ..hub.facts import lineage
from ..project import Project

MANUAL = "manual"


def _change(applied: dict) -> str:
    return f"`{applied['why']}`" if applied["id"] == MANUAL else applied["id"]


def built_from(p: Project, exp: str) -> str:
    step = lineage(p, exp)[-1]
    if not step.get("parent"):
        return "no recorded parent"
    return f"{step['parent']} + " + ", ".join(_change(a) for a in step["applied"])
