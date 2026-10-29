from __future__ import annotations

import contextlib
import json
import os
import sys
import threading
import time
from collections.abc import Iterator
from typing import Any

TARGET = "target"
EVENTS_ENV = "BYOSM_EVENTS"
JOB_ENV = "BYOSM_JOB_ID"
EVENT_PREFIX = "@event "


_WRITE = threading.Lock()


def emit(kind: str, **data: Any) -> None:
    if os.environ.get(EVENTS_ENV) != "1":
        return
    line = EVENT_PREFIX + json.dumps({"kind": kind, "at": round(time.time(), 2), **data}, default=str) + "\n"
    with _WRITE:
        sys.stdout.write(line)
        sys.stdout.flush()


def parse(line: str) -> dict[str, Any] | None:
    if not line.startswith(EVENT_PREFIX):
        return None
    try:
        data = json.loads(line[len(EVENT_PREFIX) :])
    except ValueError:
        return None
    return data if isinstance(data, dict) else None


@contextlib.contextmanager
def stage(name: str) -> Iterator[None]:
    t0 = time.time()
    emit("stage", name=name, state="running")
    try:
        yield
    except BaseException:
        emit("stage", name=name, state="failed", seconds=round(time.time() - t0, 1))
        raise
    emit("stage", name=name, state="done", seconds=round(time.time() - t0, 1))
