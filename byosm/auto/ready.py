from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

MAX_ROUNDS = 12
BUDGET_H = 6.0


def _next(name: str, tried: set[tuple], model: str | None = None) -> dict[str, Any] | None:
    from ..next_steps import suggestions

    fresh = [s for s in suggestions(name) if (model, s["id"], *s["args"]) not in tried]
    return max(fresh, key=lambda s: (bool(s.get("unblocks")), s.get("gain") or 0), default=None)


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

    p, deadline, tried = Project(name), time.time() + budget_h * 3600, set()
    for n in range(rounds):
        h = health(p, p.spec()) or {"score": 0, "ready": False, "blockers": []}
        log(f"[ready] round {n + 1}: health {h['score']}, blocked by {', '.join(h['blockers']) or 'nothing'}")
        if h["ready"]:
            return {"ready": True, "score": h["score"], "rounds": n}
        model = p.promoted if p.exists() else None
        step = _next(name, tried, model) if time.time() < deadline else None
        if step is None:
            reason = "time budget spent" if time.time() >= deadline else "no untried step left"
            return {"ready": False, "score": h["score"], "rounds": n, "stop": reason}
        tried.add((model, step["id"], *step["args"]))
        try:
            with stage(step["title"]):
                code = run(step["args"])
        except Exception as e:
            log(f"[ready] {step['title']} failed: {type(e).__name__}: {e}")
            continue
        if isinstance(code, int) and code:
            log(f"[ready] {step['title']} failed: exit code {code}")
    return {"ready": False, "rounds": rounds, "stop": f"{rounds} rounds"}
