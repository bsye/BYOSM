from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ..db.workspace import save_workspace_report, workspace_report
from .papers import Paper, fetch

SCOUT = "scout"
ROLE = "scout"
PROPOSED, ACCEPTED, REJECTED = "proposed", "accepted", "rejected"
EVERY_DAYS = 7
DAY_S = 86_400
MAX_SEEN = 5000
ARXIV_ABS = "arxiv.org/abs/"
KINDS = ["text", "audio", "image", "document", "series"]
SYSTEM = """You read recent machine-learning papers for a system that builds small specialist models (encoders of 4M to
400M parameters, small forecasters, gradient boosting) that must beat large reference models on public benchmarks.
From the papers listed, propose only techniques such a pipeline could adopt with modest compute: a training recipe,
loss, data method, model choice or prediction-time trick, with the gain the paper itself reports. Cite only papers from
the list, by their exact URL. If a technique is already in the catalogue, name its id in overlaps. Propose nothing
rather than something weak."""
ITEM = {
    "paper": {"type": "string"},
    "title": {"type": "string"},
    "effect": {"type": "string"},
    "kinds": {"type": "array", "items": {"type": "string", "enum": KINDS}},
    "gain": {"type": "string"},
    "how": {"type": "string"},
    "overlaps": {"type": "string"},
}
SCHEMA = {
    "type": "object",
    "properties": {"proposals": {"type": "array", "items": {"type": "object", "properties": ITEM}}},
}


def state(ws: Path | None) -> dict[str, Any]:
    return workspace_report(ws, SCOUT) or {"last_run": None, "seen": [], "proposals": []}


def every_days(cfg: dict) -> float:
    return float((cfg.get("scout") or {}).get("every_days", EVERY_DAYS))


def due(ws: Path | None, cfg: dict, now: float) -> bool:
    last = state(ws)["last_run"]
    return last is None or now - last >= every_days(cfg) * DAY_S


def _prompt(papers: list[Paper]) -> str:
    from ..improve.techniques import techniques

    catalogue = [{"id": t.id, "title": t.title, "effect": t.effect} for t in techniques()]
    listing = [{"url": p.url, "area": p.area, "title": p.title, "abstract": p.summary} for p in papers]
    return f"Catalogue:\n{json.dumps(catalogue)}\n\nPapers:\n{json.dumps(listing)}"


def paper_key(url: str) -> str:
    bare = url.strip().lower().removeprefix("https://").removeprefix("http://").removeprefix("www.").rstrip("/")
    head, sep, version = bare.rpartition("v")
    return head if ARXIV_ABS in bare and sep and version.isdigit() and head[-1:].isdigit() else bare


def _reason(r: dict, cited: str | None, known: set[str]) -> str | None:
    if not r.get("title"):
        return "it has no title"
    if not cited:
        return f"it cites {r.get('paper') or 'no paper'}, which this scan did not read"
    overlap = str(r.get("overlaps") or "").strip()
    return f"the catalogue already has it as {overlap}" if overlap in known else None


def _kept(raw: list[dict], papers: list[Paper], log: Callable[[str], None]) -> list[dict[str, Any]]:
    from ..improve.techniques import techniques

    urls, known = {paper_key(p.url): p.url for p in papers}, {t.id for t in techniques()}
    out = []
    for r in raw:
        cited = urls.get(paper_key(str(r.get("paper") or "")))
        if why := _reason(r, cited, known):
            log(f"  dropped {r.get('title') or 'a proposal'}: {why}")
            continue
        pid = hashlib.sha1(f"{cited}{r['title']}".encode()).hexdigest()[:10]
        fields = {k: r.get(k) for k in ITEM if k != "overlaps"} | {"paper": cited}
        out.append({**fields, "id": pid, "status": PROPOSED, "found": time.time()})
    return out


def scan(ws: Path | None, log: Callable[[str], None] = print, llm: Any = None, get: Any = None) -> list[dict]:
    from ..llm import get_llm

    st = state(ws)
    save_workspace_report(ws, SCOUT, {**st, "last_run": time.time()})
    seen = set(st["seen"])
    papers = [p for p in (fetch(log, get) if get else fetch(log)) if p.url not in seen]
    raw = (llm or get_llm(ROLE)).json(SYSTEM, _prompt(papers), SCHEMA).get("proposals", []) if papers else []
    new = _kept(raw, papers, log)
    st = {"last_run": time.time(), "seen": [*st["seen"], *(p.url for p in papers)][-MAX_SEEN:]}
    st["last_scan"] = {"read": len(papers), "proposed": len(raw), "kept": len(new)}
    save_workspace_report(ws, SCOUT, {**st, "proposals": state(ws)["proposals"] + new})
    log(f"[scout] {len(papers)} new papers read; the LLM proposed {len(raw)}, {len(new)} kept as new techniques")
    return new


def decide(ws: Path | None, pid: str, status: str) -> dict[str, Any]:
    st = state(ws)
    found = next((p for p in st["proposals"] if p["id"] == pid), None)
    if found is None:
        raise KeyError(f"no proposed technique {pid}")
    found["status"] = status
    save_workspace_report(ws, SCOUT, st)
    return found
