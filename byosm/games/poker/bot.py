from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .distill import MODEL
from .distill import REPORT as STUDENTS
from .infosets import SLOTS
from .results import against
from .table import AGREEMENT, REPORT, TOTAL, VERSUS
from .teacher import work_dir

KIND = "game"
TITLE = "Six-max hold'em bot"
DESCRIPTION = "A small student of a self-play teacher: paste a hand and get how often to play each legal move"
METRIC = "agreement with Pluribus"
SUITE = "pluribus"
BENCHMARK = "Pluribus's logged decisions"
STUDENT, TEACHER = "Student", "Teacher"
RIVAL = "its teacher"


def fetched(ws: Path) -> list[str]:
    from .sharingan import BLUEPRINTS, WEIGHTS
    from .sharingan import work_dir as blueprint_dir

    return [name for name, held in BLUEPRINTS.items() if (blueprint_dir(ws) / WEIGHTS / held.file).exists()]


def stored(ws: Path) -> tuple[dict, dict, dict]:
    from ...db.workspace import workspace_reports

    return workspace_reports(ws, STUDENTS), workspace_reports(ws, AGREEMENT), workspace_reports(ws, REPORT)


def built(ws: Path, student: str) -> str | None:
    path = work_dir(ws) / f"{student}{MODEL}"
    return datetime.fromtimestamp(path.stat().st_mtime, UTC).isoformat() if path.exists() else None


def top_move(agreements: dict, name: str) -> float | None:
    found = agreements.get(f"{AGREEMENT}{name}")
    return found[TOTAL]["top_move"] if found else None


def summaries(ws: Path) -> list[dict[str, Any]]:
    students, agreements, _ = stored(ws)
    out = []
    for s in students.values():
        teacher = top_move(agreements, s["teacher"])
        out.append(
            {
                "name": s["student"],
                "kind": KIND,
                "title": TITLE,
                "created": built(ws, s["student"]),
                "description": DESCRIPTION,
                "modality": KIND,
                "task_type": KIND,
                "labels": SLOTS,
                "metric": METRIC,
                "primary_suite": SUITE,
                "benchmark": BENCHMARK,
                "promoted": None,
                "score": top_move(agreements, s["student"]),
                "rival": {"name": RIVAL, "score": teacher} if teacher is not None else None,
                "experiments": [],
                "exported": True,
                "part_of": None,
            }
        )
    return out


def seated(key: str, r: dict, roles: dict[str, str]) -> dict[str, Any]:
    others = f"copies of the {roles[r['field']].lower()}" if VERSUS in key and r["field"] in roles else against(key, r)
    return {**r, "who": roles.get(r["strategy"], r["strategy"]), "against": others}


def detail(ws: Path, name: str) -> dict[str, Any] | None:
    students, agreements, tables = stored(ws)
    student = students.get(f"{STUDENTS}{name}")
    if not student:
        return None
    roles = {name: STUDENT, student["teacher"]: TEACHER}
    here = {k: r for k, r in tables.items() if r["strategy"] in roles or r["field"] == name}
    played = [seated(k, r, roles) for k, r in sorted(here.items())]
    found = ((role, agreements.get(f"{AGREEMENT}{n}")) for n, role in roles.items())
    agreed = [{**a, "who": role} for role, a in found if a]
    made = {"name": name, "title": TITLE, "student": student, "played": played, "agreement": agreed}
    return {**made, "blueprints": fetched(ws)}
