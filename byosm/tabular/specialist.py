from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

KIND = "risk"
MODEL = "gradient boosting"
METRIC = "AUC"


def summary(row: dict[str, Any]) -> dict[str, Any]:
    spec, result = row["spec"], row.get("result") or {}
    scores = result.get("scores") or {}
    rivals = {m: s["auc"] for m, s in scores.items() if m != MODEL}
    rival = max(rivals, key=lambda m: rivals[m]) if rivals else None
    years = spec["horizon_years"]
    return {
        "name": row["name"],
        "kind": KIND,
        "title": spec["title"],
        "created": row.get("created") or (datetime.fromtimestamp(result["at"], UTC).isoformat() if result else None),
        "description": "within a year" if years == 1 else f"within {years} years",
        "modality": "table",
        "task_type": KIND,
        "labels": 2,
        "metric": METRIC,
        "higher_is_better": True,
        "primary_suite": "backtest",
        "benchmark": "backtest",
        "promoted": MODEL if MODEL in scores else None,
        "score": scores.get(MODEL, {}).get("auc"),
        "ci": tuple(scores[MODEL]["auc_ci"]) if MODEL in scores else None,
        "rival": {"name": rival, "score": rivals[rival]} if rival else None,
        "experiments": [],
        "exported": bool(result.get("today")),
        "part_of": None,
    }
