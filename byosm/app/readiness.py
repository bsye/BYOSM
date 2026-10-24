from __future__ import annotations

from typing import Any


def interval(ci95: dict[str, float] | None) -> tuple[float, float] | None:
    return (ci95["lo"], ci95["hi"]) if ci95 else None


def readiness(
    score: float | None, rival: dict[str, Any] | None, ci: tuple[float, float] | None, lower_is_better: bool = False
) -> str:
    if score is None:
        return "unbuilt"
    theirs = (rival or {}).get("score")
    if not theirs:
        return "no_rival"
    ours = (rival or {}).get("ours", score)
    ahead = ours < theirs if lower_is_better else ours > theirs
    if not ahead:
        return "ties" if ours == theirs else "behind"
    if ci is None:
        return "no_interval"
    clear = ci[1] < theirs if lower_is_better else ci[0] > theirs
    return "ready" if clear else "ties"


def with_readiness(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {**s, "ready": readiness(s["score"], s["rival"], s.get("ci"), s.get("higher_is_better") is False)}
        for s in items
    ]
