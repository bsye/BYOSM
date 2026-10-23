from __future__ import annotations

import hashlib
import json
import time
import urllib.parse
import urllib.request
import zipfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

FOLDER = "opendata"
META = "meta.json"
CHUNK = 1 << 20
LOG_EVERY = 0.1
TIMEOUT_S = 60
TABLE_SUFFIXES = (".csv", ".tsv", ".xlsx", ".xls", ".json", ".parquet")


def _root() -> Path:
    from ..config import cache_root

    return cache_root() / FOLDER


def folder(url: str) -> Path:
    return _root() / hashlib.sha1(url.encode()).hexdigest()[:16]


def _name(url: str, headers: Any) -> str:
    disposition = headers.get("Content-Disposition") or ""
    _, _, given = disposition.partition("filename=")
    return given.strip('"; ') or Path(urllib.parse.urlparse(url).path).name or "data"


def _download(url: str, out: Path, log: Callable[[str], None], max_mb: float | None) -> Path:
    request = urllib.request.Request(url, headers={"User-Agent": "byosm"})
    with urllib.request.urlopen(request, timeout=TIMEOUT_S) as r:
        total, done, shown = int(r.headers.get("Content-Length") or 0), 0, 0.0
        if max_mb and total > max_mb * 1e6:
            raise ValueError(f"{total / 1e6:.0f} MB is over the {max_mb:g} MB limit for automatic downloads")
        path = out / _name(url, r.headers)
        with path.open("wb") as f:
            while chunk := r.read(CHUNK):
                f.write(chunk)
                done += len(chunk)
                if total and done / total - shown >= LOG_EVERY:
                    shown = done / total
                    log(f"  downloaded {done / 1e6:.0f} of {total / 1e6:.0f} MB")
    return path


def tables(out: Path) -> list[Path]:
    return sorted(p for p in out.rglob("*") if p.suffix.lower() in TABLE_SUFFIXES and p.name != META)


def fetch(url: str, log: Callable[[str], None] = print, max_mb: float | None = None) -> dict[str, Any]:
    out = folder(url)
    meta = out / META
    if meta.exists():
        log(f"  cached: {out}")
        return json.loads(meta.read_text())
    out.mkdir(parents=True, exist_ok=True)
    try:
        path = _download(url, out, log, max_mb)
    except ValueError:
        out.rmdir()
        raise
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as z:
            z.extractall(out / "unzipped")
        path.unlink()
    info = {"url": url, "fetched": time.time(), "files": [str(p.relative_to(out)) for p in tables(out)]}
    meta.write_text(json.dumps(info, indent=1))
    log(f"  saved {len(info['files'])} table file(s) in {out}")
    return info


def cached(url: str) -> dict[str, Any] | None:
    meta = folder(url) / META
    return json.loads(meta.read_text()) if meta.exists() else None


def table(url: str, file: str | None = None) -> Path:
    info = cached(url)
    if not info or not info["files"]:
        raise KeyError("not downloaded yet: download it first")
    return folder(url) / (file if file in info["files"] else info["files"][0])
