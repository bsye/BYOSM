from __future__ import annotations

from collections.abc import Callable
from datetime import date
from pathlib import Path
from typing import Any

from ..routing import ROUTE_ROLE

CANDIDATES = 4
MAX_MB = 100
BACKTEST_YEARS = 5
BACKTEST_STEP = 2
DESIGN_SYSTEM = """You design a forecast from open data tables. Pick the one table that lists entities with a date each \
started and a date each ended (empty while still alive). Name its start column, its end column, and the columns known \
when an entity starts that could predict its ending (type, sector, place, size). Never pick a column that is only filled \
or changed after the ending (a status, a dissolution or closure date, a last filing date). Also name up to three columns \
that identify an entity for a person reading the results, and a short question as the title."""


def _design_schema() -> dict[str, Any]:
    names = {"type": "array", "items": {"type": "string"}}
    keys = {
        "table": {"type": "integer"},
        "start": {"type": "string"},
        "event": {"type": "string"},
        "title": {"type": "string"},
    }
    return {"type": "object", "properties": {**keys, "features": names, "shown": names}}


def _candidates(ws: Path, searches: list[str], log: Callable[[str], None]) -> list[dict[str, Any]]:
    from ..opendata import sources
    from ..opendata.cache import fetch, remote_mb, table
    from ..opendata.catalog import search
    from ..opendata.preview import preview

    seen, out = set(), []
    for q in searches:
        for d in search(q, extra=sources.portals(ws)):
            for f in d["files"]:
                mb = remote_mb(f["url"])
                if f["url"] in seen or len(out) >= CANDIDATES or (mb and mb > MAX_MB):
                    continue
                seen.add(f["url"])
                try:
                    fetch(f["url"], log, MAX_MB)
                    look = preview(table(f["url"]))
                except (OSError, ValueError, KeyError) as e:
                    log(f"  skip {d['title']}: {type(e).__name__}")
                    continue
                out.append({"dataset": d, "file": f, "columns": look["columns"]})
    return out


def _listing(tables: list[dict[str, Any]]) -> str:
    def cols(t: dict[str, Any]) -> str:
        return "; ".join(
            f"{c['name']} ({c['missing']:.0%} empty, e.g. {', '.join(c['examples'][:2])})" for c in t["columns"]
        )

    return "\n".join(f"{i}. {t['dataset']['title']}: {cols(t)}" for i, t in enumerate(tables))


def design(prompt: str, plan: dict[str, Any], tables: list[dict[str, Any]], cfg: dict | None) -> dict[str, Any]:
    from ..llm import get_llm

    out = get_llm(ROUTE_ROLE, cfg).json(
        DESIGN_SYSTEM, f"Request: {prompt}\n\nTables:\n{_listing(tables)}", _design_schema()
    )
    chosen = tables[int(out.get("table", 0))] if 0 <= int(out.get("table", 0)) < len(tables) else tables[0]
    names = {c["name"] for c in chosen["columns"]}
    if out.get("start") not in names or out.get("event") not in names:
        raise ValueError("the LLM named start or end columns the table does not have")
    if out["start"] == out["event"]:
        raise ValueError("the LLM named the same column for start and end")
    horizon = max(1, int(plan.get("horizon_years") or 1))
    test = date.today().year - horizon
    return {
        "title": out.get("title") or prompt,
        "source": chosen["file"]["url"],
        "start": out["start"],
        "event": out["event"],
        "horizon_years": horizon,
        "test_year": test,
        "backtest": [test - i * BACKTEST_STEP for i in range(BACKTEST_YEARS)],
        "features": [c for c in out.get("features", []) if c in names and c not in (out["start"], out["event"])],
        "shown": [c for c in out.get("shown", []) if c in names],
    }


def compose(ws: Path, prompt: str, plan: dict[str, Any], cfg: dict | None, log: Callable[[str], None]) -> str:
    from .task import create, run

    log(f"  a forecast: searching open data for {plan['search']}")
    tables = _candidates(ws, plan["search"], log)
    if not tables:
        raise RuntimeError("no open data table with readable files matched; find one on the Datasets page first")
    spec = design(prompt, plan, tables, cfg)
    name = "-".join("".join(c if c.isalnum() else " " for c in spec["title"].lower()).split())[:60]
    create(ws, name, spec)
    log(f"  saved forecast {name} on {spec['source']}: {spec['start']} to {spec['event']}, features {spec['features']}")
    run(ws, name, log)
    return name
