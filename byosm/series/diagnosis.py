from __future__ import annotations

from pathlib import Path
from typing import Any

from .specialist import FIT, PREDICT, best_rival, promoted_version

SCORES = ("calib_mase", "mase", "crps")
VERSION_FIELDS = ("name", "recipe", "effort", PREDICT, FIT, "calib", "test")
DIGITS = 4


def _scores(row: dict[str, Any]) -> dict[str, float]:
    return {k: round(row[k], DIGITS) for k in SCORES if row.get(k) is not None}


def diagnose(ws: Path, domain: str) -> dict[str, Any]:
    from ..improve.diagnosis import HEALTH_FIELDS, STEP_FIELDS
    from .board import promoted_board, reports
    from .health import health
    from .steps import next_steps

    report = next((r for r in reports(ws) if r["domain"] == domain), {})
    version = promoted_version(report)
    if version is None:
        raise KeyError(f"{domain} has no promoted version yet: Keep training builds one")
    configs = promoted_board(report).get("configs", {})
    return {
        "datasets": list(configs),
        "promoted": {k: version.get(k) for k in VERSION_FIELDS},
        "rival": best_rival(report),
        "board": {c: {m: _scores(s) for m, s in row.items()} for c, row in configs.items()},
        "health": [{k: c.get(k) for k in HEALTH_FIELDS} for c in (health(report) or {}).get("checks", [])],
        "next_steps": [{k: s.get(k) for k in STEP_FIELDS} for s in next_steps(report)],
        "versions": [{k: v.get(k) for k in VERSION_FIELDS} for v in report.get("versions", [])],
    }
