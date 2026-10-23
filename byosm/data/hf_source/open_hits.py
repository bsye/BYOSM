from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .hub import FILE_MARK, OPEN_PREFIX, search_terms

TEXT = "text"
TERMS = 2
READABLE = ("CSV", "TSV", "XLSX", "XLS", "JSON", "ZIP")
ABOUT_CHARS = 280
OPEN_SHOWN = 8
AUTO_MAX_MB = 100
HEAD_TIMEOUT_S = 10


def size_mb(url: str) -> float | None:
    import urllib.request

    request = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "byosm"})
    try:
        with urllib.request.urlopen(request, timeout=HEAD_TIMEOUT_S) as r:
            length = r.headers.get("Content-Length")
    except (OSError, ValueError):
        return None
    return int(length) / 1e6 if length else None


def _small(hits: list[dict[str, Any]]) -> list[dict[str, Any]]:
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(len(hits) or 1) as pool:
        sizes = list(pool.map(lambda h: size_mb(h["id"].removeprefix(OPEN_PREFIX)), hits))
    out = []
    for h, mb in zip(hits, sizes, strict=True):
        if mb is None or mb <= AUTO_MAX_MB:
            out.append({**h, "title": h["title"] + (f", {mb:.1f} MB" if mb else "")})
    return out


def open_hits(queries: list[str], modality: str) -> list[dict[str, Any]]:
    from ...opendata.catalog import search

    if modality != TEXT or not queries:
        return []
    out: dict[str, dict[str, Any]] = {}
    for term in search_terms(queries)[:TERMS]:
        for d in search(term):
            file = next((f for f in d["files"] if f["format"] in READABLE), None)
            if file:
                out.setdefault(
                    file["url"],
                    {
                        "id": OPEN_PREFIX + file["url"],
                        "downloads": 0,
                        "title": f"{d['title']} ({d['source']}, {file['format']} file)",
                        "about": " ".join((d.get("notes") or "").split())[:ABOUT_CHARS],
                        "license": d.get("licence"),
                    },
                )
    return _small(list(out.values()))


def fetched(dataset: str, log: Callable[[str], None]) -> bool:
    from ...opendata.cache import fetch

    url = dataset.removeprefix(OPEN_PREFIX).partition(FILE_MARK)[0]
    try:
        return bool(fetch(url, log, AUTO_MAX_MB)["files"])
    except (OSError, ValueError) as e:
        log(f"  skip {url}: {type(e).__name__}: {str(e)[:120]}")
        return False
