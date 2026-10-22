from __future__ import annotations

import json
import os
import time
from typing import Any

EVENTS_ENV = "BYOSM_EVENTS"
EVENT_PREFIX = "@event "


def emit(kind: str, **data: Any) -> None:
    if os.environ.get(EVENTS_ENV) == "1":
        print(EVENT_PREFIX + json.dumps({"kind": kind, "at": round(time.time(), 2), **data}, default=str), flush=True)


def parse(line: str) -> dict[str, Any] | None:
    if not line.startswith(EVENT_PREFIX):
        return None
    try:
        data = json.loads(line[len(EVENT_PREFIX) :])
    except ValueError:
        return None
    return data if isinstance(data, dict) else None
