from __future__ import annotations

from typing import Any

from ..config import Project


def _baselines(board: dict, prim: str, field: str | None = None) -> dict[str, Any]:
    rows = {k: v[prim] for k, v in board.items() if prim in v and v[prim]["kind"] == "baseline"}
    return {k: r[field] for k, r in rows.items()} if field else rows


def _test_view(base: Any, board: dict, prim: str, mine: dict | None, best_bl: tuple | None) -> dict[str, Any]:
    from .. import evals

    return {
        "ours": mine.get("objective") if mine else None,
        "ci95": (mine or {}).get("metrics", {}).get("ci95"),
        "baselines": _baselines(board, prim, "objective"),
        "vs_best_baseline": evals.compare(base, mine, board[best_bl[0]][prim], "test") if best_bl and mine else None,
    }


def standing(name: str, margin: float) -> dict[str, Any]:
    from .. import evals
    from ..spec import TaskSpec

    p = Project(name)
    base = TaskSpec.load(p.spec_path)
    prim = base.primary_suite()
    b = p.board()
    mine = b.get(p.promoted or "", {}).get(prim)
    ours = evals.selection_value(mine)
    bls = {k: evals.selection_value(r) for k, r in _baselines(b, prim).items()}
    best_bl = max(bls.items(), key=lambda kv: kv[1]) if bls else None
    beats = ours is not None and (best_bl is None or ours > best_bl[1] + margin)
    meets = ours is not None and ours >= base.objective.target
    return {
        "suite": prim,
        "ours": ours,
        "exp": p.promoted,
        "baselines": bls,
        "best_baseline": best_bl,
        "beats": beats,
        "meets": meets,
        "won": beats and meets,
        "metric": base.objective.metric,
        "target": base.objective.target,
        "test": _test_view(base, b, prim, mine, best_bl),
    }
