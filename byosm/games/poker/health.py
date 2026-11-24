from __future__ import annotations

from pathlib import Path
from typing import Any

from ...health_score import FAIL, OK, UNKNOWN, WARN, check, report
from .bot import stored
from .distill import ABSTRACTION, MODEL
from .distill import REPORT as STUDENTS
from .results import ARENA
from .table import AGREEMENT, REPORT, TOTAL, VERSUS, key
from .teacher import work_dir

HOLDS = -100.0
MORE_HANDS = 2
SLACK = 0.02
MOST_MB = 50
CALLER = "caller"
TITLES = {
    "table": "Holds its seat against its teacher",
    "bots": "Beats simple bots",
    "rules": "Beats rule-based bots",
    "pluribus": "Matches Pluribus's moves like its teacher",
    "advice": "Answers a hand",
    "size": "Small",
}
STEPS = {
    "table": "Seat it against five copies of its teacher",
    "bots": "Seat it against five always-call bots",
    "rules": "Seat it against the arena's six rule-based bots",
    "pluribus": "Score it on Pluribus's logged moves",
    "advice": "Rebuild the student",
}


def won(r: dict) -> str:
    return f"{r['mbb_per_hand']:+.1f} ±{r['ci95']:.1f} mbb/hand"


def table(name: str, teacher: str, tables: dict) -> dict:
    seated = ["poker", "teacher-versus", name, teacher]
    why = f"What it wins against five copies of its teacher: the low end of the interval should stay above {HOLDS:g} mbb/hand."
    r = tables.get(f"{REPORT}{name}{VERSUS}{teacher}")
    if not r:
        return check("table", TITLES["table"], 3, UNKNOWN, "not seated yet", why, seated)
    if r["mbb_per_hand"] - r["ci95"] >= HOLDS:
        return check("table", TITLES["table"], 3, OK, won(r), why)
    if r["mbb_per_hand"] >= HOLDS:
        return check("table", TITLES["table"], 3, WARN, won(r), why, [*seated, "--hands", str(MORE_HANDS * r["hands"])])
    return check("table", TITLES["table"], 3, FAIL, won(r), why, ["poker", "student", "--strategy", teacher])


def bots(name: str, tables: dict) -> dict:
    why = "What it wins against five always-call bots: the whole interval should be above zero."
    r = tables.get(key(name, CALLER, False))
    if not r:
        return check(
            "bots", TITLES["bots"], 1, UNKNOWN, "not seated yet", why, ["poker", "teacher-play", "--strategy", name]
        )
    return check("bots", TITLES["bots"], 1, OK if r["mbb_per_hand"] - r["ci95"] > 0 else FAIL, won(r), why)


def rules(name: str, tables: dict) -> dict:
    why = (
        "What it wins against five copies of each of the arena's rule-based bots: every interval should be above zero."
    )
    found = {field: tables.get(f"{REPORT}{name}{VERSUS}{field}") for field in ARENA}
    if not all(found.values()):
        seated = ["poker", "arena-gauntlet", name]
        return check("rules", TITLES["rules"], 3, UNKNOWN, "not seated yet", why, seated)
    field, worst = min(found.items(), key=lambda scored: scored[1]["mbb_per_hand"] - scored[1]["ci95"])
    status = OK if worst["mbb_per_hand"] - worst["ci95"] > 0 else FAIL
    return check("rules", TITLES["rules"], 3, status, f"{won(worst)} against five {ARENA[field]} bots", why)


def pluribus(name: str, teacher: str, agreements: dict) -> dict:
    why = f"Its share of Pluribus's logged moves should be within {SLACK:.0%} of its teacher's."
    mine, theirs = (agreements.get(f"{AGREEMENT}{n}") for n in (name, teacher))
    missing = next((n for n, found in ((name, mine), (teacher, theirs)) if not found), None)
    if missing:
        scoring = ["poker", "teacher-benchmark", "--strategy", missing]
        return check("pluribus", TITLES["pluribus"], 1, UNKNOWN, "not scored yet", why, scoring)
    ours, bar = mine[TOTAL]["top_move"], theirs[TOTAL]["top_move"]
    return check(
        "pluribus", TITLES["pluribus"], 1, OK if ours >= bar - SLACK else WARN, f"{ours:.1%}, teacher {bar:.1%}", why
    )


def advice(ws: Path, name: str, teacher: str) -> dict:
    why = "Its model and its card abstraction are on disk, so a pasted hand is answered at once."
    there = all((work_dir(ws) / f"{name}{suffix}").exists() for suffix in (MODEL, ABSTRACTION))
    if there:
        return check("advice", TITLES["advice"], 2, OK, "ready", why)
    return check(
        "advice", TITLES["advice"], 2, UNKNOWN, "files missing", why, ["poker", "student", "--strategy", teacher]
    )


def size(student: dict) -> dict:
    status = OK if student["model_mb"] <= MOST_MB else WARN
    return check(
        "size", TITLES["size"], 1, status, f"{student['model_mb']} MB", f"The model should stay under {MOST_MB} MB."
    )


def checks(ws: Path, name: str) -> list[dict] | None:
    students, agreements, tables = stored(ws)
    student = students.get(f"{STUDENTS}{name}")
    if not student:
        return None
    teacher = student["teacher"]
    return [
        table(name, teacher, tables),
        bots(name, tables),
        rules(name, tables),
        pluribus(name, teacher, agreements),
        advice(ws, name, teacher),
        size(student),
    ]


def of_bot(ws: Path, name: str) -> dict[str, Any] | None:
    found = checks(ws, name)
    return report(found) if found is not None else None


def steps_of_bot(ws: Path, name: str) -> list[dict[str, Any]] | None:
    found = checks(ws, name)
    if found is None:
        return None
    return [
        {"id": c["id"], "title": STEPS[c["id"]], "why": f"{c['title']}: {c['value']}. {c['why']}", "args": c["args"]}
        for c in found
        if c["status"] != OK and c.get("args")
    ]
