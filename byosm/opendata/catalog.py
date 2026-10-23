from __future__ import annotations

import json
import urllib.parse
import urllib.request
from typing import Any

CKAN_PORTALS = {"data.gov.ie": "https://data.gov.ie", "data.gov.uk": "https://ckan.publishing.service.gov.uk"}
EU_PORTAL = "data.europa.eu"
EU_SEARCH = "https://data.europa.eu/api/hub/search/search"
EU_DATASET = "https://data.europa.eu/data/datasets/"
TABULAR = ("CSV", "TSV", "XLSX", "XLS", "JSON", "ZIP", "PARQUET")
PER_SOURCE = 10
TIMEOUT_S = 20
LANGS = ("en", "ga")


def _get(url: str) -> Any:
    request = urllib.request.Request(url, headers={"User-Agent": "byosm"})
    with urllib.request.urlopen(request, timeout=TIMEOUT_S) as r:
        return json.loads(r.read())


def _text(v: Any) -> str:
    if isinstance(v, dict):
        return str(next((v[k] for k in LANGS if v.get(k)), next(iter(v.values()), "")))
    return str(v or "")


def _tabular(fmt: str) -> bool:
    return fmt.upper().strip(".") in TABULAR


def _ckan(portal: str, query: str) -> list[dict[str, Any]]:
    q = urllib.parse.urlencode({"q": query, "rows": PER_SOURCE})
    found = _get(f"{CKAN_PORTALS[portal]}/api/3/action/package_search?{q}")["result"]["results"]
    return [
        {
            "source": portal,
            "id": p["name"],
            "title": p.get("title") or p["name"],
            "publisher": (p.get("organization") or {}).get("title", ""),
            "licence": p.get("license_title") or p.get("license_id") or "",
            "modified": p.get("metadata_modified", "")[:10],
            "url": f"https://{portal}/dataset/{p['name']}",
            "notes": (p.get("notes") or "")[:400],
            "files": [
                {
                    "name": r.get("name") or r.get("format", ""),
                    "format": (r.get("format") or "").upper(),
                    "url": r["url"],
                }
                for r in p.get("resources", [])
                if r.get("url") and _tabular(r.get("format") or "")
            ],
        }
        for p in found
    ]


def _eu_files(distributions: list[dict]) -> list[dict[str, str]]:
    out = []
    for d in distributions:
        fmt = str((d.get("format") or {}).get("id") or "")
        url = next(iter(d.get("download_url") or d.get("access_url") or []), "")
        if url and _tabular(fmt):
            out.append({"name": _text(d.get("title")) or fmt, "format": fmt.upper(), "url": url})
    return out


def _eu(query: str) -> list[dict[str, Any]]:
    q = urllib.parse.urlencode({"q": query, "limit": PER_SOURCE, "filter": "dataset"})
    found = _get(f"{EU_SEARCH}?{q}")["result"]["results"]
    return [
        {
            "source": EU_PORTAL,
            "id": r["id"],
            "title": _text(r.get("title")) or r["id"],
            "publisher": _text((r.get("publisher") or {}).get("name")),
            "licence": next(
                (_text((d.get("license") or {}).get("label")) for d in r.get("distributions", []) if d.get("license")),
                "",
            ),
            "modified": str(r.get("modified") or "")[:10],
            "url": EU_DATASET + r["id"],
            "notes": _text(r.get("description"))[:400],
            "files": _eu_files(r.get("distributions", [])),
        }
        for r in found
    ]


def search(query: str, sources: list[str] | None = None) -> list[dict[str, Any]]:
    wanted = sources or [*CKAN_PORTALS, EU_PORTAL]
    out: list[dict[str, Any]] = []
    for source in wanted:
        try:
            out += _eu(query) if source == EU_PORTAL else _ckan(source, query)
        except (OSError, ValueError, KeyError):
            continue
    return [d for d in out if d["files"]]
