from __future__ import annotations

import time
import urllib.parse
from pathlib import Path
from typing import Any

from sqlalchemy import select

from ..db import get_or_add, session
from ..db.models import WorkspaceReport
from .catalog import CKAN, CKAN_PORTALS, EU_PORTAL, SOCRATA, ckan_search
from .socrata import socrata_search

SOURCES = "open-data-sources"
PAGE = "page"
PROBES = ((CKAN, ckan_search), (SOCRATA, socrata_search))
YOU, FOUND = "you", "found"


def saved(ws: Path) -> list[dict[str, Any]]:
    with session(ws) as s:
        return s.scalar(select(WorkspaceReport.data).where(WorkspaceReport.kind == SOURCES)) or []


def listed(ws: Path) -> list[dict[str, Any]]:
    builtin = [{"name": n, "url": u, "kind": CKAN, "by": "built in"} for n, u in CKAN_PORTALS.items()]
    return [*builtin, {"name": EU_PORTAL, "url": f"https://{EU_PORTAL}", "kind": CKAN, "by": "built in"}, *saved(ws)]


def portals(ws: Path) -> list[dict[str, Any]]:
    return [s for s in saved(ws) if s["kind"] != PAGE]


def _probe(url: str) -> tuple[str, str]:
    root = urllib.parse.urlunparse(urllib.parse.urlparse(url)._replace(path="", query="", fragment=""))
    for base in dict.fromkeys((url, root)):
        for kind, probe in PROBES:
            try:
                probe(base, "", rows=1)
            except (OSError, ValueError, KeyError, TypeError):
                continue
            return kind, base
    return PAGE, url


def add(ws: Path, url: str, name: str = "", why: str = "", by: str = YOU) -> list[dict[str, Any]]:
    url = url.strip().rstrip("/")
    if not urllib.parse.urlparse(url).scheme:
        url = f"https://{url}"
    host = urllib.parse.urlparse(url).netloc
    kind, url = _probe(url)
    rows = [s for s in saved(ws) if s["url"] != url]
    entry = {"name": name or host, "url": url, "kind": kind, "why": why, "by": by, "added": time.time()}
    with session(ws) as s:
        get_or_add(s, WorkspaceReport, kind=SOURCES).data = [*rows, entry]
    return listed(ws)


def remove(ws: Path, url: str) -> list[dict[str, Any]]:
    with session(ws) as s:
        get_or_add(s, WorkspaceReport, kind=SOURCES).data = [x for x in saved(ws) if x["url"] != url]
    return listed(ws)
