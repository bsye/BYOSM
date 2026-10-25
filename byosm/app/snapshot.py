from __future__ import annotations

import hashlib
import os
import shutil
import threading
from pathlib import Path

FOLDER = "code"
SKIPPED = ("__pycache__", "static")
KEEP_DAYS = 7
DAY_S = 86_400
SEEN: dict = {}
COPYING = threading.Lock()


def _files(src: Path) -> list[Path]:
    return sorted(p for p in src.rglob("*") if p.is_file() and not any(s in p.parts for s in SKIPPED))


def _digest(src: Path) -> str:
    files = _files(src)
    stamp = tuple((p.as_posix(), p.stat().st_mtime_ns) for p in files)
    if SEEN.get("stamp") != stamp:
        digest = hashlib.sha1()
        for path in files:
            digest.update(path.relative_to(src).as_posix().encode())
            digest.update(path.read_bytes())
        SEEN.update(stamp=stamp, digest=digest.hexdigest()[:12])
    return SEEN["digest"]


def frozen() -> Path:
    import byosm

    from ..config import cache_root

    src = Path(byosm.__file__).parent
    dest = cache_root() / FOLDER / _digest(src)
    with COPYING:
        if not (dest / src.name).exists():
            staging = dest.with_name(f"{dest.name}.{os.getpid()}.staging")
            shutil.rmtree(staging, ignore_errors=True)
            shutil.copytree(src, staging / src.name, ignore=shutil.ignore_patterns(*SKIPPED))
            try:
                staging.rename(dest)
            except OSError:
                shutil.rmtree(staging, ignore_errors=True)
            _prune(dest.parent)
    return dest


def _prune(folder: Path) -> None:
    import time

    for old in folder.iterdir():
        if old.is_dir() and time.time() - old.stat().st_mtime > KEEP_DAYS * DAY_S:
            shutil.rmtree(old, ignore_errors=True)
