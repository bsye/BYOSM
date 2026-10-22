from __future__ import annotations

import contextlib
import fcntl
import time
from collections.abc import Callable, Iterator
from pathlib import Path

GPU_DEVICES = ("mps", "cuda")
SLOTS_KEY = "gpu_jobs"
DEFAULT_SLOTS = 1
LOCK_DIR = Path(".byosm") / "gpu"
WAIT_S = 5


def _slots() -> int:
    from ..config import load_config

    return max(1, int(load_config().get(SLOTS_KEY) or DEFAULT_SLOTS))


def _try(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("w")
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        handle.close()
        return None
    return handle


@contextlib.contextmanager
def gpu_slot(device: str, log: Callable[[str], None] = print) -> Iterator[None]:
    if device not in GPU_DEVICES:
        yield
        return
    from ..config import find_workspace
    from ..events import emit

    base, waited = find_workspace() / LOCK_DIR, 0.0
    while True:
        handle = next((h for i in range(_slots()) if (h := _try(base / f"slot-{i}.lock"))), None)
        if handle:
            break
        if not waited:
            log(f"  waiting for the {device} GPU: another job is training on it")
            emit("gpu", state="waiting")
        time.sleep(WAIT_S)
        waited += WAIT_S
    if waited:
        log(f"  got the GPU after {waited:.0f} s")
    try:
        yield
    finally:
        fcntl.flock(handle, fcntl.LOCK_UN)
        handle.close()
