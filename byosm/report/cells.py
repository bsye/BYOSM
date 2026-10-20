from __future__ import annotations

from ..config import Project
from ..format import fmt, interval

STALE = "†"


def num(x: float | None) -> str:
    return fmt(x, 3, "-")


def score(v: float | None, ci: dict | None) -> str:
    return num(v) if v is None or not ci else f"{num(v)} [{interval(ci, sep=', ')}]"


def folded(p: Project, exp: str) -> int:
    return p.meta(exp).get("n_test_folded") or 0


def train_rows(p: Project, exp: str) -> str:
    n = p.meta(exp).get("n_train")
    if n is None:
        return "-"
    k = folded(p, exp)
    return f"{n} (+{k} builtin test)" if k else str(n)


def fold_note(shown: bool) -> str:
    if not shown:
        return ""
    return (
        " +N builtin test = the builtin suite's held-out split was folded into training (`train.holdout_test`); "
        "the benchmark's own test split is never trained on."
    )


def stale_ids(p: Project, suite: str) -> set[str]:
    board = p.board()
    fp = board.get(p.promoted or "", {}).get(suite, {}).get("fingerprint")
    return {pid for pid, by in board.items() if fp and by.get(suite, {}).get("fingerprint", fp) != fp}


def stale_note(stale: set[str]) -> str:
    if not stale:
        return ""
    return (
        f" {STALE} = scored on a different version of this suite or its decision policy than the promoted model; "
        "rerun `byosm eval` to refresh."
    )
