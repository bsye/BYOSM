from __future__ import annotations

from pathlib import Path
from typing import Any

from ..health_score import OK

PRETRAINING = "pretraining-v2"
FORECAST_ROUNDS = 2


def keep_training(domain: str) -> list[str]:
    return ["series", "run", domain, "--until-rival"]


def write_method(domain: str) -> list[str]:
    return ["lab", "write", domain]


def pretraining_step(report: dict[str, Any]) -> dict[str, Any] | None:
    from .escalation import tried
    from .patch import pretrained
    from .recipe import V2

    if tried(report, V2):
        return None
    found = {"id": PRETRAINING, "title": "Pretrain on the v2 recipe", "model": report["promoted"]}
    if pretrained(V2) is None:
        why = "No forecaster was trained on the v2 recipe yet. Pretrains its base once, for every domain."
        return {**found, "why": why, "args": ["series", "pretrain", "--recipe", V2.name]}
    domain = report["domain"]
    why = f"No version of {domain} was trained on the v2 base. Adds versions until one beats the best rival; v2 comes once more training stopped helping."
    return {**found, "why": why, "args": [*keep_training(domain), "--rounds", str(FORECAST_ROUNDS)]}


def of_domain(ws: Path, name: str) -> list[dict[str, Any]] | None:
    from .board import report_of

    found = report_of(ws, name)
    return next_steps(ws, found) if found and found["at"] else None


def _why(check: dict[str, Any]) -> str:
    return f"{check['title']} is not ok: {check['value']}. {check['why']}"


def next_steps(ws: Path, report: dict[str, Any]) -> list[dict[str, Any]]:
    from ..lab.owners import Forecaster
    from ..lab.steps import lab_steps
    from .health import health

    checks = (health(report) or {}).get("checks", [])
    fixes = [
        {"id": c["id"], "title": f"Try {', '.join(c['techniques'])}", "why": _why(c), "args": c["args"]}
        for c in checks
        if c["status"] != OK and c.get("args")
    ]
    return [*fixes, *filter(None, [pretraining_step(report)]), *lab_steps(Forecaster(ws, report["domain"]))]
