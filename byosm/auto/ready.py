from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

MAX_ROUNDS = 12
BUDGET_H = 6.0


def _next(name: str, tried: set[tuple], model: str | None = None, blockers: list[str] = ()) -> dict[str, Any] | None:
    from ..next_steps import suggestions

    fresh = [s for s in suggestions(name) if (model, s["id"], *s["args"]) not in tried]
    clears = {s["id"]: bool(set(s.get("unblocks") or []) & set(blockers)) for s in fresh}
    return max(fresh, key=lambda s: (clears[s["id"]], bool(s.get("unblocks")), s.get("gain") or 0), default=None)


def _report_move(step: dict[str, Any], before: dict[str, str], h: dict[str, Any], log: Callable[[str], None]) -> None:
    from ..next_steps import STEP_CHECKS

    after = {c["id"]: c["status"] for c in h.get("checks", [])}
    for cid in STEP_CHECKS.get(step["id"], ()):
        if cid in before:
            change = "unchanged" if after.get(cid) == before[cid] else f"now {after.get(cid)}"
            log(f"[ready] {step['title']}: check {cid} was {before[cid]}, {change}")


def until_ready(
    name: str,
    run: Callable[[list[str]], Any],
    log: Callable[[str], None] = print,
    rounds: int = MAX_ROUNDS,
    budget_h: float = BUDGET_H,
) -> dict[str, Any]:
    from ..events import stage
    from ..health import health
    from ..project import Project

    p, deadline, tried, moved = Project(name), time.time() + budget_h * 3600, set(), []
    for n in range(rounds):
        h = health(p, p.spec()) or {"score": 0, "ready": False, "blockers": []}
        if moved:
            _report_move(*moved.pop(), h, log)
        log(f"[ready] round {n + 1}: health {h['score']}, blocked by {', '.join(h['blockers']) or 'nothing'}")
        if h["ready"]:
            return {"ready": True, "score": h["score"], "rounds": n}
        model = p.promoted if p.exists() else None
        step = _next(name, tried, model, h["blockers"]) if time.time() < deadline else None
        if step is None:
            reason = "time budget spent" if time.time() >= deadline else "no untried step left"
            return {"ready": False, "score": h["score"], "rounds": n, "stop": reason}
        tried.add((model, step["id"], *step["args"]))
        moved.append((step, {c["id"]: c["status"] for c in h.get("checks", [])}))
        try:
            with stage(step["title"]):
                code = run(step["args"])
        except Exception as e:
            log(f"[ready] {step['title']} failed: {type(e).__name__}: {e}")
            continue
        if isinstance(code, int) and code:
            log(f"[ready] {step['title']} failed: exit code {code}")
    return {"ready": False, "rounds": rounds, "stop": f"{rounds} rounds"}
