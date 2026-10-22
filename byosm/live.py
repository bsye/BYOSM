from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Callable
from datetime import timedelta
from pathlib import Path
from typing import Any

import numpy as np

from .config import secret
from .project import Project

URL_KEY, API_KEY_KEY, SECRET_KEY, PUBLIC_URL_KEY = (
    "LIVEKIT_URL",
    "LIVEKIT_API_KEY",
    "LIVEKIT_API_SECRET",
    "LIVEKIT_PUBLIC_URL",
)
DEV_URL, DEV_KEY, DEV_SECRET = "ws://localhost:7880", "devkey", "secret"
TOPIC = "vad"
LISTENER = "byosm-listener"
SAMPLE_RATE = 16000
CHUNK_MS = 32
SILERO_WINDOW = 512
SILERO_THRESHOLD = 0.5
INT16_SCALE = 32768.0
TOKEN_TTL = timedelta(hours=1)
MAX_SESSION_S = 15 * 60
ALONE_GRACE_S = 20


def settings(ws: Path | None = None) -> dict[str, str]:
    url = secret(URL_KEY, ws) or DEV_URL
    return {
        "url": url,
        "public_url": secret(PUBLIC_URL_KEY, ws) or url,
        "key": secret(API_KEY_KEY, ws) or DEV_KEY,
        "secret": secret(SECRET_KEY, ws) or DEV_SECRET,
    }


def token(identity: str, room: str, ws: Path | None = None) -> str:
    from livekit import api

    s = settings(ws)
    grants = api.VideoGrants(room_join=True, room=room)
    return (
        api.AccessToken(s["key"], s["secret"]).with_identity(identity).with_grants(grants).with_ttl(TOKEN_TTL).to_jwt()
    )


class Scorers:
    def __init__(self, bundle: Path):
        from . import rt
        from .predictors.audio import silero_model

        self.ours = rt.load(bundle).stream()
        self.silero = silero_model()
        self.pending = np.zeros(0, np.float32)
        self.silero_p = 0.0

    def push(self, chunk: np.ndarray) -> dict[str, Any]:
        t0 = time.perf_counter()
        ours = self.ours.step(chunk)
        t1 = time.perf_counter()
        self.pending = np.concatenate([self.pending, chunk])
        while len(self.pending) >= SILERO_WINDOW:
            self.silero_p = float(self.silero(np.ascontiguousarray(self.pending[:SILERO_WINDOW])[None]))
            self.pending = self.pending[SILERO_WINDOW:]
        t2 = time.perf_counter()
        probs = ours["probs"]
        return {
            "ours": round(max(probs), 4) if probs else None,
            "ours_speaking": ours["speaking"],
            "silero": round(self.silero_p, 4),
            "silero_speaking": self.silero_p >= SILERO_THRESHOLD,
            "ours_ms": round((t1 - t0) * 1000, 3),
            "silero_ms": round((t2 - t1) * 1000, 3),
        }


async def listen(name: str, room_name: str, ws: Path | None = None, log: Callable[[str], None] = print) -> None:
    from livekit import rtc

    from .export.bundle import exported_bundle

    scorers = Scorers(exported_bundle(Project(name, ws).root, name))
    room = rtc.Room()
    done = asyncio.Event()

    async def pump(track: Any) -> None:
        stream = rtc.AudioStream(track, sample_rate=SAMPLE_RATE, num_channels=1, frame_size_ms=CHUNK_MS)
        async for ev in stream:
            x = np.frombuffer(bytes(ev.frame.data), np.int16).astype(np.float32) / INT16_SCALE
            out = {"t": round(time.time(), 3), **scorers.push(x)}
            await room.local_participant.publish_data(json.dumps(out), reliable=False, topic=TOPIC)

    def on_track(track: Any, _pub: Any, who: Any) -> None:
        if track.kind == rtc.TrackKind.KIND_AUDIO:
            log(f"  listening to {who.identity}")
            asyncio.ensure_future(pump(track))

    def on_left(who: Any) -> None:
        log(f"  {who.identity} left")
        if not room.remote_participants:
            done.set()

    room.on("track_subscribed", on_track)
    room.on("participant_disconnected", on_left)
    s = settings(ws)
    await room.connect(s["url"], token(LISTENER, room_name, ws))
    log(f"  joined room {room_name} on {s['url']}")
    try:
        await asyncio.wait_for(_until_alone_or_done(room, done), MAX_SESSION_S)
    except TimeoutError:
        log(f"  session hit its {MAX_SESSION_S // 60} min limit")
    finally:
        await room.disconnect()


async def _until_alone_or_done(room: Any, done: asyncio.Event) -> None:
    await asyncio.sleep(ALONE_GRACE_S)
    if not room.remote_participants:
        return
    await done.wait()
