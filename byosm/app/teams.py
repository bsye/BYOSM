from __future__ import annotations

from pathlib import Path
from typing import Any

OURS = "system"
TEAM_METRIC = "exact_frame"
TEAM_METRIC_LABEL = "exact frame match"
MIN_FOLDED_MEMBERS = 2


def team_names(ws: Path) -> list[str]:
    from ..system.spec import system_names

    return system_names(ws)


def members(ws: Path) -> dict[str, str]:
    from ..system import SystemProject

    nodes = {t: SystemProject(t, ws).spec().nodes.values() for t in team_names(ws)}
    return {n.specialist: t for t, ns in nodes.items() if len(ns) >= MIN_FOLDED_MEMBERS for n in ns}


def _best_rival(suite: dict) -> dict[str, Any] | None:
    rivals = {k: v["test"][TEAM_METRIC] for k, v in suite["rivals"].items() if k != OURS}
    if not rivals:
        return None
    name = max(rivals, key=lambda k: rivals[k])
    return {"name": name, "score": rivals[name]}


def team_summary(ws: Path, name: str) -> dict[str, Any]:
    from .catalog import system

    t = system(ws, name)
    board = t["board"] or {}
    suite_name, suite = next(iter(board.items()), (None, None))
    ours = suite["rivals"].get(OURS) if suite else None
    return {
        "name": name,
        "kind": "team",
        "title": t["spec"]["title"],
        "created": t["spec"]["created"],
        "description": t["spec"]["description"],
        "modality": "text",
        "task_type": "team",
        "labels": 0,
        "metric": TEAM_METRIC_LABEL,
        "primary_suite": suite_name,
        "benchmark": suite_name,
        "promoted": None,
        "score": ours["test"][TEAM_METRIC] if ours else None,
        "rival": _best_rival(suite) if suite else None,
        "experiments": [],
        "exported": t["exported"],
        "members": [n["specialist"] for n in t["spec"]["nodes"].values()],
        "part_of": None,
    }
