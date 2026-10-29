from __future__ import annotations

from collections.abc import Callable, Hashable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

_SNAPSHOT: ContextVar[dict[Hashable, Any] | None] = ContextVar("snapshot", default=None)


@contextmanager
def snapshot() -> Iterator[None]:
    token = _SNAPSHOT.set({})
    try:
        yield
    finally:
        _SNAPSHOT.reset(token)


def remembered(key: Hashable, read: Callable[[], Any]) -> Any:
    found = _SNAPSHOT.get()
    if found is None:
        return read()
    if key not in found:
        found[key] = read()
    return found[key]
