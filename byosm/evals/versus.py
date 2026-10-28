from __future__ import annotations

from typing import Any

from .selection import SURE_LEAD, compare

SPLITS = ("calib", "test")


def versus(p: Any, a: str, b: str) -> dict[str, Any]:
    spec, board = p.spec(), p.board()
    unknown = [m for m in (a, b) if m not in board]
    if unknown:
        raise KeyError(f"not on the board of {p.name}: {', '.join(unknown)}")
    rows = []
    for suite in (e.name for e in spec.evals):
        ra, rb = board[a].get(suite), board[b].get(suite)
        if ra and rb:
            rows.append({"suite": suite, **{split: compare(spec, rb, ra, split) for split in SPLITS}})
    return {"a": a, "b": b, "sure": SURE_LEAD, "suites": rows}
