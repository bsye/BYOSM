from __future__ import annotations

from collections.abc import Callable

RULES: list[Callable[[dict], list[dict]]] = []


def rule(fn: Callable[[dict], list[dict]]) -> Callable[[dict], list[dict]]:
    RULES.append(fn)
    return fn


def gap(dg: dict, suite: str) -> tuple[float, tuple[str, float] | None]:
    r = dg["suites"].get(suite, {})
    bb = r.get("best_baseline")
    return ((bb[1] - r["objective"]) if bb else 0.0), bb


def rule_proposals(dg: dict) -> list[dict]:
    seen, out = set(), []
    for r in RULES:
        for pr in r(dg):
            if pr["id"] not in seen:
                seen.add(pr["id"])
                out.append(pr)
    return out
