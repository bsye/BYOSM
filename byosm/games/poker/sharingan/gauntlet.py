from __future__ import annotations

import math
from pathlib import Path

from ..sealed import Log
from . import blueprint
from .engine import BENCH, loading, policy, run, stored

REPORT = "poker-gauntlet:"
DEALS = 20_000
SEATS = 6
TABLE = f"{SEATS}-max"
HEADER = "=="
RAW = "raw"
CORRECTED = "aivat"
PUBLISHED = {
    "sharingan-2": {
        "nit": (125.0, 35.0),
        "tag": (303.0, 56.0),
        "lag": (933.0, 85.0),
        "maniac": (1622.0, 99.0),
        "station": (2572.0, 118.0),
        "random": (4004.0, 197.0),
    },
}


def estimate(words: list[str], label: str) -> dict[str, float] | None:
    if label not in words:
        return None
    at = words.index(label)
    return {"mbb_per_hand": float(words[at + 1]), "ci95": float(words[at + 3])}


def parsed(said: list[str]) -> dict[str, dict]:
    found, counted = {}, False
    for line in said:
        words = line.split()
        if line.startswith(HEADER):
            counted = TABLE in line
        elif counted and words[:1] == ["vs"]:
            found[words[1]] = {RAW: estimate(words, RAW), CORRECTED: estimate(words, CORRECTED)}
    if not found:
        raise RuntimeError("the gauntlet printed no six-player result")
    return found


def reproduced(ours: dict[str, float], theirs: tuple[float, float]) -> bool:
    return abs(ours["mbb_per_hand"] - theirs[0]) <= math.hypot(ours["ci95"], theirs[1])


def compared(name: str, found: dict[str, dict]) -> dict[str, dict]:
    return {
        field: {
            **scored,
            "published": list(PUBLISHED[name][field]),
            "reproduced": reproduced(scored[CORRECTED], PUBLISHED[name][field]),
        }
        for field, scored in found.items()
        if field in PUBLISHED.get(name, {}) and scored[CORRECTED]
    }


def gauntlet(ws: Path, name: str, deals: int, log: Log) -> dict:
    from ....db.workspace import save_workspace_report

    found = blueprint(name)
    stored(ws, found)
    said = run(ws, [BENCH, "gauntlet", "--hero", policy(found), *loading(found), "--deals", str(deals)], log)
    fields = parsed(said)
    checks = compared(name, fields)
    made = {"blueprint": name, "deals": deals, "hands": deals * SEATS, "fields": fields, "checks": checks}
    save_workspace_report(ws, f"{REPORT}{name}", made)
    for field, check in checks.items():
        ours, theirs = check[CORRECTED], check["published"]
        verdict = "reproduced" if check["reproduced"] else "NOT reproduced"
        log(
            f"against {field}: {ours['mbb_per_hand']:+.1f} ±{ours['ci95']:.1f} mbb/hand here, "
            f"{theirs[0]:+.0f} ±{theirs[1]:.0f} published: {verdict}"
        )
    return made
