from __future__ import annotations

from collections.abc import Callable

PLATEAU_SPREAD = 0.01
RULES: list[Callable[[dict], list[dict]]] = []


def rule(fn: Callable[[dict], list[dict]]) -> Callable[[dict], list[dict]]:
    RULES.append(fn)
    return fn


def gap(dg: dict, suite: str) -> tuple[float, tuple[str, float] | None]:
    r = dg["suites"].get(suite, {})
    bb = r.get("best_baseline")
    return ((bb[1] - r["objective"]) if bb else 0.0), bb


def plateaued(dg: dict) -> bool:
    curve = dg["curve"]
    tail = curve[-max(2, len(curve) // 3) :]
    return len(curve) >= 2 and not dg["late_best"] and max(tail) - min(tail) < PLATEAU_SPREAD


def refine_hurt(dg: dict) -> bool:
    return dg.get("refine_gain") is not None and dg["refine_gain"] <= 0


def unique(props: list[dict]) -> list[dict]:
    out: dict[str, dict] = {}
    for pr in props:
        out.setdefault(pr["id"], pr)
    return list(out.values())


def rule_proposals(dg: dict) -> list[dict]:
    return unique([pr for r in RULES for pr in r(dg)])
