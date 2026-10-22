from __future__ import annotations

import base64
import json
import threading
import time
from collections.abc import Callable
from typing import Protocol, TypeVar

from .limits import is_limit

T = TypeVar("T")

TRANSIENT_MARKERS = ("rate", "overload", "timeout", "connect")


class LLMError(RuntimeError):
    pass


_usage = threading.local()


def report_usage(tokens_in: int | None, tokens_out: int | None) -> None:
    _usage.value = {"tokens_in": tokens_in, "tokens_out": tokens_out}


def take_usage() -> dict[str, int | None]:
    value = getattr(_usage, "value", None) or {"tokens_in": None, "tokens_out": None}
    _usage.value = None
    return value


class LLM(Protocol):
    name: str

    def json(self, system: str, prompt: str, schema: dict, images: list[bytes] | None = None) -> dict: ...


def _strict(schema: dict) -> dict:
    s = dict(schema)
    if s.get("type") == "object":
        props = {k: _strict(v) for k, v in s.get("properties", {}).items()}
        s["properties"] = props
        s["required"] = list(props)
        s["additionalProperties"] = False
    if s.get("type") == "array" and "items" in s:
        s["items"] = _strict(s["items"])
    return s


def _fenced_body(text: str) -> str | None:
    start = text.find("```")
    if start < 0:
        return None
    body = text[start + 3 :]
    if body.startswith("json"):
        body = body[4:]
    body = body.lstrip()
    end = body.find("```")
    if end < 0:
        return None
    return body[:end]


def _parse_json(text: str) -> dict:
    text = text.strip()
    fenced = _fenced_body(text)
    if fenced is not None:
        text = fenced
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start >= 0 and end > start:
            return json.loads(text[start : end + 1])
        raise


def _is_transient(e: Exception) -> bool:
    msg = str(e).lower()
    return any(marker in msg for marker in TRANSIENT_MARKERS)


def _retry(fn: Callable[[], T], tries: int = 4) -> T:
    for i in range(tries):
        try:
            return fn()
        except (json.JSONDecodeError, LLMError) as e:
            if i == tries - 1 or is_limit(e):
                raise
        except Exception as e:
            if i == tries - 1 or not _is_transient(e):
                raise
            time.sleep(2 ** (i + 1))
    raise AssertionError("unreachable")


def _b64(img: bytes) -> tuple[str, str]:
    return ("image/png" if img[:4] == b"\x89PNG" else "image/jpeg"), base64.b64encode(img).decode()
