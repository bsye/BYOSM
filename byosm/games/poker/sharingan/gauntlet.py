from __future__ import annotations

import math
from pathlib import Path

from ..sealed import Log
from .arena import HOUSE, SEATS, played

REPORT = "poker-gauntlet:"
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


def reproduced(ours: dict, theirs: tuple[float, float]) -> bool:
    return abs(ours["mbb_per_hand"] - theirs[0]) <= math.hypot(ours["ci95"], theirs[1])


def compared(name: str, fields: dict[str, dict]) -> dict[str, dict]:
    known = PUBLISHED.get(name, {})
    return {
        field: {"published": list(known[field]), "reproduced": reproduced(scored, known[field])}
        for field, scored in fields.items()
        if field in known
    }


def told(field: str, scored: dict, check: dict | None) -> str:
    said = f"against five {field}: {scored['mbb_per_hand']:+.1f} ±{scored['ci95']:.1f} mbb/hand"
    if not check:
        return said
    verdict = "reproduced" if check["reproduced"] else "NOT reproduced"
    return f"{said}, {check['published'][0]:+.0f} ±{check['published'][1]:.0f} published: {verdict}"


def gauntlet(ws: Path, name: str, deals: int, log: Log) -> dict:
    from ....db.workspace import save_workspace_report

    fields = {field: played(ws, name, field, deals, log) for field in HOUSE}
    checks = compared(name, fields)
    made = {"player": name, "deals": deals, "hands": deals * SEATS, "fields": fields, "checks": checks}
    save_workspace_report(ws, f"{REPORT}{name}", made)
    for field, scored in fields.items():
        log(told(field, scored, checks.get(field)))
    return made
