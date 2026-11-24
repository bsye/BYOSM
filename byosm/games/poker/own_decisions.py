from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from .sealed import Log
from .sharingan.decisions import RAISES, Asked

ALL_IN = "all"
REFUSED = "refused"
REPORT_EVERY = 10_000


def kind(move: str) -> str:
    word = move.split()[0]
    return RAISES if word in (RAISES, ALL_IN) else word


def response(advice: dict[str, Any]) -> dict[str, Any]:
    weights: dict[str, float] = defaultdict(float)
    for move in advice["moves"]:
        weights[kind(move["move"])] += move["probability"]
    top = max(advice["moves"], key=lambda move: move["probability"])
    probs = [{"type": found, "p": weight} for found, weight in weights.items()]
    return {"action": {"type": kind(top["move"])}, "probs": probs, "fallback_reason": None}


def answers(ws: Path, name: str, source: str, found: list[Asked], log: Log) -> Iterator[dict[str, Any]]:
    from .advisor import Advisor

    advisor = Advisor.start(ws, None, name, log)
    try:
        for done, logged in enumerate(found, start=1):
            try:
                yield {"key": logged.key, "ok": True, "response": response(advisor.advise(logged.logged))}
            except ValueError:
                yield {"key": logged.key, "ok": False, "error": {"code": REFUSED}}
            if done % REPORT_EVERY == 0:
                log(f"progress: {done:,} of {len(found):,} decisions of {source}")
    finally:
        advisor.close()
