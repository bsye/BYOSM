from __future__ import annotations

import socket
import uuid
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from fastapi import APIRouter, HTTPException

from .errors import missing

BROWSER_IDENTITY = "you"
REACH_TIMEOUT_S = 1.0
ROOM_SUFFIX_CHARS = 6


def reachable(url: str) -> bool:
    parts = urlsplit(url)
    default = 443 if parts.scheme in ("wss", "https") else 80
    try:
        with socket.create_connection((parts.hostname or "localhost", parts.port or default), REACH_TIMEOUT_S):
            return True
    except OSError:
        return False


def _session(ws: Path, name: str, runner: Any) -> dict[str, Any]:
    from ..config import Project
    from ..export.bundle import exported_bundle
    from ..live import settings, token
    from ..spec import TaskSpec

    p = Project(name, ws)
    if not p.exists():
        raise KeyError(f"no specialist {name!r}")
    if TaskSpec.load(p.spec_path).modality != "audio":
        raise ValueError("live microphone tests are for audio specialists")
    exported_bundle(p.root, name)
    s = settings(ws)
    if not reachable(s["url"]):
        raise HTTPException(503, f"no LiveKit server at {s['url']}: run livekit-server --dev, or set LIVEKIT_URL")
    room = f"{name}-{uuid.uuid4().hex[:ROOM_SUFFIX_CHARS]}"
    job = runner.submit(["listen", name, "--room", room])
    return {"url": s["public_url"], "token": token(BROWSER_IDENTITY, room, ws), "room": room, "job": job.id}


def live_router(ws: Path, runner: Any) -> APIRouter:
    router = APIRouter()

    @router.post("/api/specialists/{name}/live")
    def live(name: str) -> dict:
        return missing(_session, ws, name, runner)

    return router
