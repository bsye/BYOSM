from __future__ import annotations

from typing import Any

from .grouped import request_accuracy

REQUESTS_METRIC = "exact requests"


def drop(mine: dict[str, Any], other: dict[str, Any], metric: str) -> tuple[float, str] | None:
    a, b = request_accuracy(mine.get("metrics") or {}), request_accuracy(other.get("metrics") or {})
    if a is not None and b is not None:
        return a - b, REQUESTS_METRIC
    if mine.get("objective") is None or other.get("objective") is None:
        return None
    return mine["objective"] - other["objective"], metric


def worst_drop(mine: dict[str, Any], others: dict[str, dict[str, Any]], metric: str) -> tuple[str, float, str] | None:
    found = {s: d for s, r in others.items() if (d := drop(mine, r, metric)) is not None}
    if not found:
        return None
    suite = max(found, key=lambda s: found[s][0])
    return suite, *found[suite]
