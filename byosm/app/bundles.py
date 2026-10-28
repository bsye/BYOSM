from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

LOADED_BUNDLES = 8


@lru_cache(maxsize=LOADED_BUNDLES)
def _bundle(path: str, stamp: float) -> Any:
    from ..rt import load

    return load(Path(path))


def bundle(ws: Path, name: str) -> Any:
    path = ws / "specialists" / name / "export"
    manifest = path / "manifest.json"
    if not manifest.exists():
        raise FileNotFoundError(f"{name} is not exported: run the export job first")
    return _bundle(str(path), manifest.stat().st_mtime)
