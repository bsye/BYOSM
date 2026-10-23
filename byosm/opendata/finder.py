from __future__ import annotations

import time
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import Any

from sqlalchemy import select

from ..db import get_or_add, session
from ..db.models import WorkspaceReport
from . import sources
from .catalog import TABULAR, TIMEOUT_S, search

REQUESTS = "open-data-requests"
KEPT_REQUESTS = 20
MAX_QUERIES = 4
RUNNING, DONE, FAILED = "running", "done", "failed"
WEB = "web"
MIN_WORD = 4
STEM = 5
COMMON = {
    "with",
    "their",
    "that",
    "which",
    "from",
    "data",
    "dataset",
    "table",
    "tables",
    "list",
    "about",
    "each",
    "they",
}
SYSTEM = f"""You find open data tables to build a prediction model. Search the web for public, downloadable tables \
(formats: {", ".join(TABULAR)}) that answer the request: government open data portals, statistics offices, registers, \
research repositories. Return three things. queries: always 2 to {MAX_QUERIES} short keyword searches of 1 to 3 words to run on the open data \
portals listed. datasets: tables you found on the web, each with a title, the publisher, the page that describes it, the \
licence, one line on why it fits, and its direct file links (name, format, url) that you checked download a file. \
portals: open data portals worth searching for this kind of request that are not listed yet, each with a name, its base \
url and why. Never invent a link."""


def _schema() -> dict[str, Any]:
    text = {"type": "string"}
    file = {"type": "object", "properties": {"name": text, "format": text, "url": text}}
    keys = ("title", "publisher", "page", "licence", "why")
    dataset = {"type": "object", "properties": {**{k: text for k in keys}, "files": {"type": "array", "items": file}}}
    portal = {"type": "object", "properties": {"name": text, "url": text, "why": text}}
    arrays = {"queries": text, "datasets": dataset, "portals": portal}
    return {"type": "object", "properties": {k: {"type": "array", "items": v} for k, v in arrays.items()}}


def requests(ws: Path) -> list[dict[str, Any]]:
    with session(ws) as s:
        return s.scalar(select(WorkspaceReport.data).where(WorkspaceReport.kind == REQUESTS)) or []


def _save(ws: Path, entry: dict[str, Any]) -> None:
    rest = [r for r in requests(ws) if r["request"] != entry["request"]]
    with session(ws) as s:
        get_or_add(s, WorkspaceReport, kind=REQUESTS).data = [entry, *rest][:KEPT_REQUESTS]


def forget(ws: Path, request: str) -> list[dict[str, Any]]:
    with session(ws) as s:
        get_or_add(s, WorkspaceReport, kind=REQUESTS).data = [r for r in requests(ws) if r["request"] != request]
    return requests(ws)


def _answers(url: str) -> bool:
    request = urllib.request.Request(url, headers={"User-Agent": "byosm", "Range": "bytes=0-0"})
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT_S) as r:
            return r.status < 400
    except (OSError, ValueError):
        return False


def _from_web(d: dict[str, Any]) -> dict[str, Any]:
    files = [
        {**f, "format": str(f.get("format", "")).upper()} for f in d.get("files", []) if _answers(f.get("url", ""))
    ]
    keys = ("title", "publisher", "licence", "why")
    return {
        "source": WEB,
        "id": d.get("page") or d.get("title", ""),
        "url": d.get("page", ""),
        "modified": "",
        **{k: d.get(k, "") for k in keys},
        "files": files,
    }


def _stems(text: str) -> set[str]:
    words = "".join(c if c.isalnum() else " " for c in text.lower()).split()
    return {w[:STEM] for w in words if len(w) >= MIN_WORD and w not in COMMON}


def _keywords(request: str) -> str:
    words = "".join(c if c.isalnum() else " " for c in request.lower()).split()
    return " ".join(w for w in words if len(w) >= MIN_WORD and w not in COMMON)


def relevant(found: list[dict[str, Any]], *asked: str) -> list[dict[str, Any]]:
    wanted = set().union(*map(_stems, asked))
    return [d for d in found if wanted & _stems(f"{d['title']} {d.get('notes', '')}")]


def _unseen(have: list[dict[str, Any]], found: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen = {f["url"] for d in have for f in d["files"]}
    fresh = []
    for d in found:
        if not {f["url"] for f in d["files"]} <= seen:
            fresh.append(d)
            seen |= {f["url"] for f in d["files"]}
    return fresh


def _ask(ws: Path, request: str, log: Callable[[str], None]) -> dict[str, Any]:
    from ..config import load_config
    from ..data.hf_source.research import RESEARCH_ROLE, research_config
    from ..llm import get_llm

    cfg = load_config(ws)
    conf = research_config(cfg)
    if not conf:
        raise RuntimeError("no LLM with web search found: install Claude Code or set llm.research in byosm.yaml")
    llm = get_llm(RESEARCH_ROLE, {**cfg, "llm": {**cfg["llm"], RESEARCH_ROLE: conf}})
    known = "\n".join(f"- {s['name']}: {s['url']}" for s in sources.listed(ws))
    log(f"  searching the web with {llm.name}")
    return llm.json(SYSTEM, f"Request: {request}\n\nPortals listed:\n{known}", _schema())


def find(ws: Path, request: str, log: Callable[[str], None] = print) -> dict[str, Any]:
    entry = {"request": request, "at": time.time(), "status": RUNNING, "queries": [], "datasets": [], "portals": []}
    _save(ws, entry)
    try:
        plan = _ask(ws, request, log)
        for p in plan.get("portals", []):
            if p.get("url"):
                sources.add(ws, p["url"], p.get("name", ""), p.get("why", ""), sources.FOUND)
                entry["portals"].append(p)
        web = [d for d in map(_from_web, plan.get("datasets", [])) if d["files"]]
        log(f"  the web search named {len(plan.get('datasets', []))} tables, {len(web)} with a file that downloads")
        entry["datasets"] = web
        _save(ws, entry)
        for q in (plan.get("queries") or [_keywords(request)])[:MAX_QUERIES]:
            fresh = _unseen(entry["datasets"], relevant(search(q, extra=sources.portals(ws)), request, q))
            log(f"  portals: {len(fresh)} new tables for {q!r}")
            entry["queries"].append(q)
            entry["datasets"] += fresh
            _save(ws, entry)
        entry["status"] = DONE
    except Exception:
        entry["status"] = FAILED
        raise
    finally:
        _save(ws, entry)
    return entry
