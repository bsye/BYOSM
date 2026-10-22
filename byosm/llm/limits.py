from __future__ import annotations

import datetime as dt
import time
from collections.abc import Callable
from typing import Any

LIMIT_MARKERS = ("usage limit", "spend limit", "limit reached", "hit your", "credit balance", "insufficient_quota")
RESETS = "resets"
SECTION = "on_limit"
WAIT, LOCAL, FAIL = "wait", "local", "fail"
MODES = (WAIT, LOCAL, FAIL)
POLL_S = 60
BACKOFF_MIN = (5, 10, 20, 30, 60)
MAX_WAIT_H = 12
RESET_SLACK_S = 60
LOCAL_MODEL = "qwen3.8:27b"
LOCAL_URL = "http://localhost:11434/v1"
LOCAL_KEY = "ollama"


def is_limit(e: BaseException) -> bool:
    msg = str(e).lower()
    return any(m in msg for m in LIMIT_MARKERS)


def _clock(word: str) -> tuple[int, int] | None:
    w = word.strip().lower()
    pm, am = w.endswith("pm"), w.endswith("am")
    w = w[:-2] if pm or am else w
    hour, _, minute = w.partition(":")
    if not hour.isdecimal() or (minute and not minute.isdecimal()):
        return None
    h = int(hour) % 12 + (12 if pm else 0) if pm or am else int(hour)
    return (h, int(minute or 0)) if h < 24 else None


def reset_at(message: str, now: float) -> float | None:
    _, found, rest = message.lower().partition(RESETS)
    words = rest.split()
    hm = _clock(words[0]) if found and words else None
    if hm is None:
        return None
    today = dt.datetime.fromtimestamp(now)
    at = today.replace(hour=hm[0], minute=hm[1], second=0, microsecond=0)
    if at.timestamp() <= now:
        at += dt.timedelta(days=1)
    return at.timestamp() + RESET_SLACK_S


def settings() -> dict[str, Any]:
    from ..config import load_config

    return {"mode": WAIT, **(load_config().get(SECTION) or {})}


def local_llm_config() -> dict[str, Any]:
    own = settings().get(LOCAL) or {}
    return {"provider": "openai", "base_url": LOCAL_URL, "api_key": LOCAL_KEY, "model": LOCAL_MODEL, **own}


def wait_out(error: BaseException, attempt: int, started: float, notify: Callable[[float], None]) -> str:
    now = time.time()
    if now - started > MAX_WAIT_H * 3600:
        return FAIL
    until = reset_at(str(error), now) or now + 60 * BACKOFF_MIN[min(attempt, len(BACKOFF_MIN) - 1)]
    notify(until)
    while time.time() < until:
        current = settings()
        if current["mode"] != WAIT or float(current.get("retry_now") or 0) > now:
            return current["mode"] if current["mode"] != WAIT else WAIT
        time.sleep(min(POLL_S, max(1.0, until - time.time())))
    return WAIT
