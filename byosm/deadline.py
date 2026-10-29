from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar

_DEADLINE: ContextVar[float | None] = ContextVar("deadline", default=None)


class BudgetSpentError(RuntimeError):
    pass


@contextmanager
def until(at: float) -> Iterator[None]:
    token = _DEADLINE.set(at)
    try:
        yield
    finally:
        _DEADLINE.reset(token)


def ensure_time_left(what: str) -> None:
    at = _DEADLINE.get()
    if at is not None and time.time() >= at:
        raise BudgetSpentError(f"time budget spent, so {what} was not started")
