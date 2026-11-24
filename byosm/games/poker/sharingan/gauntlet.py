from __future__ import annotations

import math
from pathlib import Path

from ..sealed import Log
from ..table import REPORT, VERSUS
from .arena import HOUSE, versus

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


def checks(tables: dict[str, dict], name: str) -> dict[str, dict]:
    found = {}
    for field, theirs in PUBLISHED.get(name, {}).items():
        ours = tables.get(f"{REPORT}{name}{VERSUS}{field}")
        if ours:
            found[field] = {"ours": ours, "published": list(theirs), "reproduced": reproduced(ours, theirs)}
    return found


def told(field: str, scored: dict, check: dict | None) -> str:
    said = f"against five {field}: {scored['mbb_per_hand']:+.1f} ±{scored['ci95']:.1f} mbb/hand"
    if not check:
        return said
    verdict = "reproduced" if check["reproduced"] else "NOT reproduced"
    return f"{said}, {check['published'][0]:+.0f} ±{check['published'][1]:.0f} published: {verdict}"


def gauntlet(ws: Path, name: str, deals: int, log: Log) -> dict[str, dict]:
    from ....db.workspace import workspace_reports

    fields = {field: versus(ws, name, field, deals, log) for field in HOUSE}
    checked = checks(workspace_reports(ws, REPORT), name)
    for field, scored in fields.items():
        log(told(field, scored, checked.get(field)))
    return fields
