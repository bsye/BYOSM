from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from .proposals import state
from .routing import methods

LAB_STATUS = "lab"
CLASSIFIER_WIN, FORECASTER_WIN, NO_WIN = "sure lead", "promoted", "no lead"


def _ran(ws: Path | None, proposal: dict[str, Any]) -> list[dict[str, Any]]:
    return [m for m in methods(ws, proposal) if m["result"]]


def tried(ws: Path | None, proposal: dict[str, Any]) -> bool:
    return bool(_ran(ws, proposal))


def _outcome(m: dict[str, Any], forecasters: set[str]) -> dict[str, Any]:
    won = bool(m["result"].get("won"))
    kind = FORECASTER_WIN if m["owner"] in forecasters else CLASSIFIER_WIN
    return {"owner": m["owner"], "method": m["name"], "result": kind if won else NO_WIN}


def _row(p: dict[str, Any], ran: list[dict[str, Any]], forecasters: set[str]) -> dict[str, Any]:
    from ..improve.techniques import published

    outcomes = [_outcome(m, forecasters) for m in ran]
    return {
        "id": f"scout-{p['id']}",
        "title": p["title"],
        "effect": p["effect"],
        "status": LAB_STATUS,
        "modalities": p.get("kinds") or [],
        "evidence": [p["paper"]],
        "evidence_dates": [published(p["paper"])],
        "added": time.strftime("%Y-%m-%d", time.localtime(p["found"])),
        "target_titles": [],
        "tries": len(ran),
        "wins": sum(o["result"] == CLASSIFIER_WIN for o in outcomes),
        "gpu_hours": 0.0,
        "specialists": len({m["owner"] for m in ran}),
        "calib_effect": {"measured": 0, "mean": None, "lo": None, "hi": None, "left_out": len(ran), "kinds": []},
        "outcomes": outcomes,
    }


def lab_families(ws: Path | None) -> list[dict[str, Any]]:
    from ..series.board import reports

    forecasters = {r["domain"] for r in reports(ws) if r["at"]}
    rows = [(p, _ran(ws, p)) for p in state(ws)["proposals"]]
    return [_row(p, ran, forecasters) for p, ran in rows if ran]
