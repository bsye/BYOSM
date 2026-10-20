from __future__ import annotations

import json

from ..config import Project

STALE = "†"


def num(x: float | None) -> str:
    return "-" if x is None else f"{x:.3f}"


def score(v: float | None, ci: dict | None) -> str:
    return num(v) if v is None or not ci else f"{v:.3f} [{ci['lo']:.3f}, {ci['hi']:.3f}]"


def params_cell(n: int | None) -> str:
    return f"{n / 1e6:.2f}M" if n else "-"


def cpu_cell(r: dict) -> str:
    if r.get("stream_ms"):
        return f"{r['stream_ms']:.3f}/{r['chunk_ms']:.0f}ms"
    return f"{r['latency_ms']:.2f}ms" if r.get("latency_ms") else "-"


def _meta(p: Project, exp: str) -> dict:
    f = p.exp_dir(exp) / "meta.json"
    return json.loads(f.read_text()) if f.exists() else {}


def folded(p: Project, exp: str) -> int:
    return _meta(p, exp).get("n_test_folded") or 0


def train_rows(p: Project, exp: str) -> str:
    n = _meta(p, exp).get("n_train")
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
