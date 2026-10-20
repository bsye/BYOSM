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


def unique(props: list[dict]) -> list[dict]:
    out: dict[str, dict] = {}
    for pr in props:
        out.setdefault(pr["id"], pr)
    return list(out.values())


def rule_proposals(dg: dict) -> list[dict]:
    return unique([pr for r in RULES for pr in r(dg)])
