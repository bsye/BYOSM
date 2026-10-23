from __future__ import annotations

import contextlib
import fcntl
import time
from collections.abc import Callable, Iterator
from pathlib import Path

GPU_DEVICES = ("mps", "cuda")
SLOTS_KEY = "gpu_jobs"
MAX_AUTO_SLOTS = 2
CORES_PER_TRAINING = 8
GB_PER_TRAINING = 16
MPS_SHARE = 0.9
LOCK_DIR = Path(".byosm") / "gpu"
WAIT_S = 5


def auto_slots() -> int:
    import os

    import psutil

    from ..machine import UNIFIED_MEMORY_SHARE

    usable_gb = psutil.virtual_memory().total * UNIFIED_MEMORY_SHARE / 1024**3
    fits = min((os.cpu_count() or 1) // CORES_PER_TRAINING, int(usable_gb // GB_PER_TRAINING))
    return max(1, min(MAX_AUTO_SLOTS, fits))


def slots() -> int:
    from ..config import load_config

    return max(1, int(load_config().get(SLOTS_KEY) or auto_slots()))


def free_gb() -> float:
    import psutil

    return psutil.virtual_memory().available / 1024**3


def _roomy() -> bool:
    return free_gb() >= GB_PER_TRAINING


def _try(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("w")
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        handle.close()
        return None
    return handle


def _cap_memory(device: str, log: Callable[[str], None]) -> None:
    import torch

    if device != "mps" or not torch.backends.mps.is_available():
        return

    share = MPS_SHARE / slots()
    torch.mps.set_per_process_memory_fraction(share)
    log(f"  this training may use {share:.0%} of the GPU's {torch.mps.recommended_max_memory() / 1024**3:.0f} GB")


@contextlib.contextmanager
def gpu_slot(device: str, log: Callable[[str], None] = print) -> Iterator[None]:
    if device not in GPU_DEVICES:
        yield
        return
    from ..config import find_workspace
    from ..events import emit

    base, waited = find_workspace() / LOCK_DIR, 0.0
    while True:
        handle = next((h for i in range(slots()) if (h := _try(base / f"slot-{i}.lock"))), None) if _roomy() else None
        if handle:
            break
        if not waited:
            log(
                f"  waiting for the {device} GPU: another job is training on it, or less than {GB_PER_TRAINING} GB is free"
            )
            emit("gpu", state="waiting")
        time.sleep(WAIT_S)
        waited += WAIT_S
    if waited:
        log(f"  got the GPU after {waited:.0f} s")
    _cap_memory(device, log)
    try:
        yield
    finally:
        fcntl.flock(handle, fcntl.LOCK_UN)
        handle.close()


def busy_slots(ws: Path) -> int:
    held = 0
    for path in (ws / LOCK_DIR).glob("slot-*.lock"):
        handle = _try(path)
        if handle is None:
            held += 1
        else:
            fcntl.flock(handle, fcntl.LOCK_UN)
            handle.close()
    return held
