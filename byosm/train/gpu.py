from __future__ import annotations

import contextlib
import fcntl
import json
import os
import time
from collections.abc import Callable, Iterator
from pathlib import Path

from ..config import STATE_DIR
from .gpu_memory import GB_PER_TRAINING, free_gb, need_gb, needed_gb, remember, total_gb
from .gpu_queue import first, join

GPU_DEVICES = ("mps", "cuda")
SLOTS_KEY = "gpu_jobs"
MAX_AUTO_SLOTS = 2
CORES_PER_TRAINING = 8
MPS_SHARE = 0.9
MIN_SHARE = 0.25
HOLDER = ".holder"
SHARE_ENV = "BYOSM_GPU_SHARE"
ADMISSION_LOCK = "admission.lock"
JOB_ENV = "BYOSM_JOB_ID"
LOCK_DIR = STATE_DIR / "gpu"
WAIT_S = 5
HEARTBEAT_S = 300


def auto_slots() -> int:
    import psutil

    from ..machine import UNIFIED_MEMORY_SHARE

    usable_gb = psutil.virtual_memory().total * UNIFIED_MEMORY_SHARE / 1024**3
    fits = min((os.cpu_count() or 1) // CORES_PER_TRAINING, int(usable_gb // GB_PER_TRAINING))
    return max(1, min(MAX_AUTO_SLOTS, fits))


def slots() -> int:
    from ..config import load_config

    return max(1, int(load_config().get(SLOTS_KEY) or auto_slots()))


def _may_start(device: str, ws: Path) -> bool:
    return device == "cuda" or busy_slots(ws) == 0 or free_gb() >= needed_gb()


def _try(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("w")
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        handle.close()
        return None
    return handle


def held_share(ws: Path) -> float:
    return sum(h.get("share", MPS_SHARE / slots()) for h in holders(ws))


def _reserved(path: Path) -> dict:
    if not path.exists():
        return {}
    text, holder = path.read_text().strip(), path.with_suffix(HOLDER)
    held = json.loads(text) if text.startswith("{") else {"share": float(text or 0)}
    return {**held, **(json.loads(holder.read_text()) if holder.exists() else {})}


def holders(ws: Path) -> list[dict]:
    out = []
    for i in range(slots()):
        lock = ws / LOCK_DIR / f"slot-{i}.lock"
        handle = _try(lock) if lock.exists() else None
        if handle is not None:
            fcntl.flock(handle, fcntl.LOCK_UN)
            handle.close()
        elif lock.exists():
            out.append({"slot": i, **_reserved(ws / LOCK_DIR / f"slot-{i}.share")})
    return out


def _share_left(device: str, ws: Path) -> float:
    if device != "mps":
        return MPS_SHARE
    return max(MPS_SHARE - held_share(ws), min(MPS_SHARE, free_gb() / total_gb()))


def _cap_memory(device: str, log: Callable[[str], None], share: float | None = None) -> None:
    import torch

    if device != "mps" or not torch.backends.mps.is_available():
        return

    share = share or float(os.environ.get(SHARE_ENV) or MPS_SHARE / slots())
    torch.mps.set_per_process_memory_fraction(share)
    log(f"  this training may use {share:.0%} of the GPU's {torch.mps.recommended_max_memory() / 1024**3:.0f} GB")


@contextlib.contextmanager
def _admission(base: Path) -> Iterator[None]:
    base.mkdir(parents=True, exist_ok=True)
    with (base / ADMISSION_LOCK).open("w") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


@contextlib.contextmanager
def gpu_slot(
    device: str, log: Callable[[str], None] = print, fallback: str | None = None, key: str | None = None
) -> Iterator[str]:
    if device not in GPU_DEVICES:
        yield device
        return
    from ..config import find_workspace
    from ..events import emit

    ws, waited = find_workspace(), 0.0
    base = ws / LOCK_DIR
    enough = max(MIN_SHARE, need_gb(base, key) / total_gb())
    ticket = None if fallback else join(base, enough)
    while True:
        with _admission(base):
            share = _share_left(device, ws)
            roomy = share >= enough and _may_start(device, ws) and (ticket is None or first(ticket, share))
            taken = (
                next(((i, h) for i in range(slots()) if (h := _try(base / f"slot-{i}.lock"))), None) if roomy else None
            )
            if taken:
                index, handle = taken
                (base / f"slot-{index}.share").write_text(f"{share:.3f}")
                holder = {"job": os.environ.get(JOB_ENV), "since": time.time()}
                (base / f"slot-{index}{HOLDER}").write_text(json.dumps(holder))
                break
        if fallback:
            log(f"  the {device} GPU is busy: training on the {fallback} instead")
            yield fallback
            return
        if not waited:
            log(
                f"  waiting for the {device} GPU: {len(holders(ws))} training(s) on it, {free_gb():.0f} GB free, {max(needed_gb(), need_gb(base, key)):.0f} GB needed"
            )
        if waited % HEARTBEAT_S < WAIT_S:
            emit("gpu", state="waiting", waited=round(waited))
        time.sleep(WAIT_S)
        waited += WAIT_S
    if ticket:
        ticket.unlink(missing_ok=True)
    if waited:
        log(f"  got the GPU after {waited:.0f} s")
    os.environ[SHARE_ENV] = f"{share:.3f}"
    _cap_memory(device, log, share)
    ran_out = None
    try:
        yield device
    except RuntimeError as e:
        ran_out = share * total_gb() if "out of memory" in str(e) else None
        raise
    finally:
        remember(base, key, ran_out)
        (base / f"slot-{index}.share").unlink(missing_ok=True)
        (base / f"slot-{index}{HOLDER}").unlink(missing_ok=True)
        fcntl.flock(handle, fcntl.LOCK_UN)
        handle.close()


def busy_slots(ws: Path) -> int:
    return len(holders(ws))
