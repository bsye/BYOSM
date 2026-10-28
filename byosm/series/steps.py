from __future__ import annotations

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


def next_steps(report: dict[str, Any]) -> list[dict[str, Any]]:
    from .health import health

    checks = (health(report) or {}).get("checks", [])
    fixes = [
        {"id": c["id"], "title": f"Try {', '.join(c['techniques'])}", "why": c["why"], "args": c["args"]}
        for c in checks
        if c["status"] != OK and c.get("args")
    ]
    return [*fixes, *filter(None, [pretraining_step(report)])]
