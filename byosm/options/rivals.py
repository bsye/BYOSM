from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .pairs import OPTION_MARK, OTHERS_MARK

ROLE = "baseline"
SAMPLE = 200
SYSTEM = """You choose which of the offered tools a request should call. Answer with the exact names of every tool the \
request needs, including the same tool for several items; answer an empty list when no offered tool fits."""


def _split(rows: list[dict[str, str]]) -> tuple[str, list[str]]:
    situation = rows[0]["text"].partition(OPTION_MARK)[0]
    return situation, [r["text"].partition(OPTION_MARK)[2].partition(OTHERS_MARK)[0] for r in rows]


def llm_chooser(cfg: dict | None, role: str = ROLE) -> Callable[[list[dict[str, str]]], set[int]]:
    from ..llm import get_llm

    llm = get_llm(role, cfg)
    schema = {"type": "object", "properties": {"tools": {"type": "array", "items": {"type": "string"}}}}

    def choose(rows: list[dict[str, str]]) -> set[int]:
        situation, options = _split(rows)
        names = [o.partition(":")[0].strip() for o in options]
        listing = "\n".join(f"- {o}" for o in options)
        picked = llm.json(SYSTEM, f"Request: {situation}\n\nTools:\n{listing}", schema).get("tools") or []
        return {names.index(n) for n in picked if n in names}

    return choose


def rival_scores(
    table: Any, cfg: dict | None, chance: Callable[[list[str]], list[float]], sample: int = SAMPLE, role: str = ROLE
) -> dict[str, Any]:
    from ..llm import parallel_calls, pmap, role_model
    from .select import THRESHOLD, decide, evaluate, sampled

    found = sampled(table, sample)
    choose = llm_chooser(cfg, role)
    picks = dict(zip(found, pmap(lambda k: choose(found[k]), found, parallel_calls(cfg)), strict=True))
    model = evaluate(found, lambda _, rows: decide(chance([r["text"] for r in rows]), THRESHOLD))
    rival = evaluate(found, lambda key, _: picks[key])
    return {"sample": len(found), "model": model, "rivals": {role_model(role, cfg): rival}}
