from __future__ import annotations

import json
import os
from pathlib import Path

from .detached import alive

LOCK_PATH = ".byosm/app.lock"


class WorkspaceBusyError(RuntimeError):
    pass


def serving(ws: Path) -> dict | None:
    path = ws / LOCK_PATH
    held = json.loads(path.read_text()) if path.exists() else None
    return held if held and alive(held["pid"]) else None


def claim(ws: Path, port: int | None) -> Path:
    path = ws / LOCK_PATH
    held = serving(ws)
    if held and held["pid"] != os.getpid():
        raise WorkspaceBusyError(
            f"another byosm app (pid {held['pid']}, port {held.get('port')}) already serves this workspace; "
            "open it, or stop it first"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"pid": os.getpid(), "port": port}))
    return path


def release(path: Path) -> None:
    held = json.loads(path.read_text()) if path.exists() else None
    if held and held["pid"] == os.getpid():
        path.unlink()
