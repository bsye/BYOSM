from __future__ import annotations

from typing import Any

OK, WARN, FAIL, UNKNOWN = "ok", "warn", "fail", "unknown"
CREDIT = {OK: 1.0, WARN: 0.5, FAIL: 0.0, UNKNOWN: 0.0}
READY_SCORE = 80
ADVISORY = {"rival", "size"}
DROP_OK, DROP_WARN = 0.05, 0.15
CHECK_TITLES = {
    "rival": "Beats the best rival",
    "target": "Meets the target",
    "overfit": "No overfitting",
    "robust": "Holds on new inputs",
    "real": "Tested on real inputs",
    "data": "Enough data",
    "certainty": "Precise score",
    "seeds": "Seed noise measured",
    "weakest": "No weak label",
    "speed": "Fast enough",
    "size": "Smaller than the rival",
    "export": "Exported",
}


def check(cid: str, title: str, weight: int, status: str, value: str, why: str, args: list[str] | None = None):
    return {"id": cid, "title": title, "weight": weight, "status": status, "value": value, "why": why, "args": args}


def graded(v: float, ok: float, warn: float, higher_is_better: bool = True) -> str:
    if not higher_is_better:
        v, ok, warn = -v, -ok, -warn
    return OK if v >= ok else WARN if v >= warn else FAIL


def report(cs: list[dict]) -> dict[str, Any] | None:
    if not cs:
        return None
    score = round(100 * sum(c["weight"] * CREDIT[c["status"]] for c in cs) / sum(c["weight"] for c in cs))
    blockers = [c["title"] for c in cs if c["status"] in (FAIL, UNKNOWN) and c["id"] not in ADVISORY]
    return {
        "score": score,
        "ready": score >= READY_SCORE and not blockers,
        "ready_score": READY_SCORE,
        "blockers": blockers,
        "checks": cs,
    }
