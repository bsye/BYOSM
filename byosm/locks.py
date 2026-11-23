from __future__ import annotations

import contextlib
import fcntl
from collections.abc import Callable, Iterator
from pathlib import Path


@contextlib.contextmanager
def exclusive(path: Path, waiting: Callable[[], None] | None = None) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            if waiting:
                waiting()
            fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)
