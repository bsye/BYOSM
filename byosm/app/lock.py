from __future__ import annotations

import json
import os
from pathlib import Path

LOCK_PATH = ".byosm/app.lock"


class WorkspaceBusyError(RuntimeError):
    pass


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def claim(ws: Path, port: int | None) -> Path:
    path = ws / LOCK_PATH
    held = json.loads(path.read_text()) if path.exists() else None
    if held and held["pid"] != os.getpid() and _alive(held["pid"]):
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
