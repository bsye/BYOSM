from __future__ import annotations

from pathlib import Path
from typing import Any

from sqlalchemy import select

from ..db import Experiment, Specialist, session
from .prices import bare, price
from .spend import usage

SECTION = "costs"
DEFAULTS = {
    "watts": 60,
    "usd_per_kwh": 0.30,
    "compare": [
        "claude-haiku-5-5",
        "claude-sonnet-5-5",
        "claude-opus-5-5",
        "gpt-5.4-mini",
        "gpt-5.6",
        "gemini-3.8-flash",
    ],
}
JOULES_PER_KWH = 3.6e6
MILLION = 1_000_000
LLM_RIVALS = ("llm", "llm:")


def settings(ws: Path) -> dict[str, Any]:
    from ..config import load_config

    return {**DEFAULTS, **(load_config(ws).get(SECTION) or {})}


def _energy_usd(seconds: float, cfg: dict[str, Any]) -> float:
    return seconds * cfg["watts"] / JOULES_PER_KWH * cfg["usd_per_kwh"]


def _train_seconds(ws: Path) -> dict[str, float]:
    with session(ws) as s:
        rows = s.execute(
            select(Specialist.name, Experiment.meta).join(Experiment, Experiment.specialist_id == Specialist.id)
        )
        out: dict[str, float] = {}
        for name, meta in rows:
            out[name] = out.get(name, 0.0) + float((meta or {}).get("train_seconds") or 0)
    return out


def _rival_items(board: dict, prim: str) -> int:
    return sum(
        int(r.get("n_test") or 0) + int(r.get("n_calib") or 0)
        for pid, by in board.items()
        if pid == LLM_RIVALS[0] or pid.startswith(LLM_RIVALS[1])
        for r in by.values()
    )


def _per_item(rival: dict[str, Any], items: int) -> dict[str, float] | None:
    if not items or not rival["calls"]:
        return None
    return {"tokens_in": rival["tokens_in"] / items, "tokens_out": rival["tokens_out"] / items}


def _compare(
    per_item: dict[str, float] | None, build_usd: float, ours: float, models: list[str]
) -> list[dict[str, Any]]:
    out = []
    for m in models:
        p = price(m)
        if p is None or per_item is None:
            continue
        item = per_item["tokens_in"] * p[0] + per_item["tokens_out"] * p[1]
        saving = item - ours
        out.append(
            {
                "model": bare(m),
                "per_item_usd": item,
                "per_million_usd": item * MILLION,
                "break_even_items": round(build_usd / saving) if saving > 0 else None,
            }
        )
    return out


def economics(ws: Path) -> list[dict[str, Any]]:
    from ..project import Project, list_projects

    cfg, spent, trained = settings(ws), usage(ws), _train_seconds(ws)
    out = []
    for name in list_projects(ws):
        p = Project(name, ws)
        spec, board = p.spec(), p.board()
        prim = spec.primary_suite()
        mine = board.get(p.promoted or "", {}).get(prim, {})
        own = spent.get(name)
        if own is None and not trained.get(name):
            continue
        llm_usd = own["build"]["usd"] if own else 0.0
        compute_usd = _energy_usd(trained.get(name, 0.0), cfg)
        ours = _energy_usd(float(mine.get("latency_ms") or 0) / 1000, cfg)
        per_item = _per_item(own["rival"], _rival_items(board, prim)) if own else None
        out.append(
            {
                "name": name,
                "title": spec.title or name,
                "build": {
                    "llm_usd": llm_usd,
                    "llm_calls": own["build"]["calls"] if own else 0,
                    "train_hours": trained.get(name, 0.0) / 3600,
                    "compute_usd": compute_usd,
                },
                "evaluate_usd": own["evaluate"]["usd"] if own else 0.0,
                "per_item": {
                    **(per_item or {}),
                    "measured": per_item is not None,
                    "ours_usd": ours,
                    "latency_ms": mine.get("latency_ms"),
                },
                "compare": _compare(per_item, llm_usd + compute_usd, ours, cfg["compare"]),
            }
        )
    return out
