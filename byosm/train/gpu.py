from __future__ import annotations

import contextlib
import fcntl
import json
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

from ..config import STATE_DIR
from ..spec import TaskSpec

GPU_DEVICES = ("mps", "cuda")
SLOTS_KEY = "gpu_jobs"
MAX_AUTO_SLOTS = 2
CORES_PER_TRAINING = 8
GB_PER_TRAINING = 16
MPS_SHARE = 0.9
OOM_RETRIES = 3
MIN_BATCH = 4
LOCK_DIR = STATE_DIR / "gpu"
WAIT_S = 5
HEARTBEAT_S = 300


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
        if waited % HEARTBEAT_S < WAIT_S:
            emit("gpu", state="waiting", waited=round(waited))
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


def out_of_memory(e: RuntimeError) -> bool:
    return "out of memory" in str(e).lower()


BATCHES_FILE = "batches.json"
RETRY_IN_PROCESS = ("cuda",)


def _batches(ws: Path) -> dict[str, int]:
    path = ws / LOCK_DIR / BATCHES_FILE
    return json.loads(path.read_text()) if path.exists() else {}


def _remember(ws: Path, key: str, size: int) -> None:
    path = ws / LOCK_DIR / BATCHES_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({**_batches(ws), key: size}, indent=1))


def _with_batch(spec: TaskSpec, size: int) -> TaskSpec:
    return spec.model_copy(update={"train": spec.train.model_copy(update={"batch_size": size})})


def batch_key(spec: TaskSpec) -> str:
    return f"{spec.modality}|{spec.backbone}"


def shrink_on_oom(
    spec: TaskSpec, train: Callable[[TaskSpec], tuple[dict, Any]], log: Callable[[str], None], device: str
):
    from ..config import find_workspace

    ws, key = find_workspace(), batch_key(spec)
    known = _batches(ws).get(key)
    if known and known < spec.train.batch_size:
        log(f"  batch {known}: {key} ran out of GPU memory at a larger batch before")
        spec = _with_batch(spec, known)
    for _ in range(OOM_RETRIES + 1):
        try:
            meta, rest = train(spec)
            return {**meta, "batch_size": spec.train.batch_size}, rest
        except RuntimeError as e:
            size = spec.train.batch_size // 2
            if not out_of_memory(e) or size < MIN_BATCH:
                raise
            _remember(ws, key, size)
            if device not in RETRY_IN_PROCESS:
                raise RuntimeError(
                    f"out of GPU memory at batch {spec.train.batch_size}; the next run starts at batch {size}"
                ) from e
            log(f"  out of GPU memory at batch {spec.train.batch_size}: retrying at batch {size}")
            spec = _with_batch(spec, size)
    raise RuntimeError("out of GPU memory at every batch size tried")
