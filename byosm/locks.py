from __future__ import annotations

import contextlib
import fcntl
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import IO

POLL_SECONDS = 0.5
REPORT_SECONDS = 600


def taken(handle: IO[str]) -> bool:
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return False
    return True


@contextlib.contextmanager
def exclusive(path: Path, waiting: Callable[[], None] | None = None) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        said = None
        while not taken(handle):
            now = time.monotonic()
            if waiting and (said is None or now - said >= REPORT_SECONDS):
                waiting()
                said = now
            time.sleep(POLL_SECONDS)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)
