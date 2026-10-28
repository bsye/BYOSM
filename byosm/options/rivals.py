from __future__ import annotations

from collections.abc import Callable
from typing import Any

ROLE = "baseline"
SAMPLE = 200
SYSTEM = """You choose which of the offered tools a request should call. Each item shows the request with one offered \
tool, written the way the specialist reads it. Answer with the numbers of every item whose tool the request needs; \
answer an empty list when no offered tool fits."""
SCHEMA = {"type": "object", "properties": {"items": {"type": "array", "items": {"type": "integer"}}}}


def listing(rows: list[dict[str, str]]) -> str:
    return "\n\n".join(f"Item {i}:\n{r['text']}" for i, r in enumerate(rows, 1))


def llm_chooser(cfg: dict | None, role: str = ROLE) -> Callable[[list[dict[str, str]]], set[int]]:
    from ..llm import get_llm

    llm = get_llm(role, cfg)

    def choose(rows: list[dict[str, str]]) -> set[int]:
        picked = llm.json(SYSTEM, listing(rows), SCHEMA).get("items") or []
        return {n - 1 for n in picked if isinstance(n, int) and 0 < n <= len(rows)}

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
