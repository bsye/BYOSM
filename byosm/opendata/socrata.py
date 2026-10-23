from __future__ import annotations

import urllib.parse
from typing import Any

PER_SOURCE = 10
TABLE = "dataset"


def _get(url: str) -> Any:
    from .catalog import _get as get

    return get(url)


def socrata_search(base: str, query: str, rows: int = PER_SOURCE) -> list[dict[str, Any]]:
    q = urllib.parse.urlencode({"q": query, "limit": rows, "only": TABLE})
    return _get(f"{base.rstrip('/')}/api/catalog/v1?{q}")["results"]


def socrata(portal: str, query: str, base: str) -> list[dict[str, Any]]:
    root = base.rstrip("/")
    out = []
    for r in socrata_search(base, query):
        res, meta = r["resource"], r.get("metadata") or {}
        out.append(
            {
                "source": portal,
                "id": res["id"],
                "title": res.get("name") or res["id"],
                "publisher": res.get("attribution") or meta.get("domain", ""),
                "licence": meta.get("license", ""),
                "modified": str(res.get("data_updated_at") or "")[:10],
                "url": r.get("permalink") or f"{root}/d/{res['id']}",
                "notes": (res.get("description") or "")[:400],
                "files": [
                    {
                        "name": res.get("name") or res["id"],
                        "format": "CSV",
                        "url": f"{root}/api/views/{res['id']}/rows.csv?accessType=DOWNLOAD",
                    }
                ],
            }
        )
    return out
