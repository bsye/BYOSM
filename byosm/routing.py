from __future__ import annotations

from collections.abc import Callable
from typing import Any

FORECAST, POKER, SPECIALIST = "forecast", "poker", "specialist"
ROUTE_ROLE = "spec"
ROUTE_SYSTEM = """Decide what a request asks for. "forecast" means it asks whether entities in a register (companies, \
licences, permits, vehicles, buildings) will end, close or lapse within some years, so it needs a table with a start date \
and an end date per entity. "poker" means it asks for decisions in no-limit hold'em poker: which move to make in a hand. \
"specialist" means anything else: classifying or tagging text, audio, images or documents. \
For a forecast also give the horizon in years and two or three short keyword searches for open data portals."""
POKER_NAME = "poker-move"
Log = Callable[[str], None]


def _schema() -> dict[str, Any]:
    kinds = {"type": "string", "enum": [FORECAST, POKER, SPECIALIST]}
    search = {"type": "array", "items": {"type": "string"}}
    return {"type": "object", "properties": {"kind": kinds, "horizon_years": {"type": "integer"}, "search": search}}


def route(prompt: str, cfg: dict | None) -> dict[str, Any]:
    from .llm import get_llm

    return get_llm(ROUTE_ROLE, cfg).json(ROUTE_SYSTEM, prompt, _schema())


def routed(prompt: str, log: Log) -> tuple[str, str | None]:
    from .config import find_workspace, load_config
    from .project import free_name

    cfg, ws = load_config(), find_workspace()
    plan = route(prompt, cfg)
    kind = plan.get("kind")
    if kind == FORECAST and plan.get("search"):
        from .tabular.compose import compose

        return FORECAST, compose(ws, prompt, plan, cfg, log)
    if kind == POKER:
        from .games.poker.create import create

        return POKER, create(ws, free_name(POKER_NAME), prompt, log).name
    return SPECIALIST, None
