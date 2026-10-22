from __future__ import annotations

import shutil
import subprocess
from collections.abc import Callable
from pathlib import Path

from .server import STATIC_DIR

UI_DIR = Path(__file__).parents[2] / "ui"
SOURCE_DIRS = ("src",)
SOURCE_FILES = ("index.html", "package.json", "vite.config.ts")


def _newest_source() -> float:
    files = [f for d in SOURCE_DIRS for f in (UI_DIR / d).rglob("*") if f.is_file()]
    files += [UI_DIR / f for f in SOURCE_FILES if (UI_DIR / f).exists()]
    return max((f.stat().st_mtime for f in files), default=0.0)


def stale() -> bool:
    built = STATIC_DIR / "index.html"
    return UI_DIR.exists() and (not built.exists() or built.stat().st_mtime < _newest_source())


def ensure_built(log: Callable[[str], None]) -> None:
    if not stale():
        return
    npm = shutil.which("npm")
    if not npm:
        log(f"the UI in {STATIC_DIR} is older than {UI_DIR}; install npm to rebuild it")
        return
    log("the UI changed since the last build: rebuilding it")
    subprocess.run([npm, "run", "build"], cwd=UI_DIR, check=True, capture_output=True)
