from __future__ import annotations

import os
from pathlib import Path

GB = 1024**3
LOCAL_CALL_GB = 2.0
JOB_GB = 3.0
REMOTE_PARALLEL = 64


def free_gb() -> float:
    import psutil

    return psutil.virtual_memory().available / GB


def local_parallel() -> int:
    return max(1, int(free_gb() // LOCAL_CALL_GB))


def max_jobs() -> int:
    return os.cpu_count() or 1


def can_start_job(running: int) -> bool:
    return running == 0 or (running < max_jobs() and free_gb() >= JOB_GB)


STALL_KEY = "stall_minutes"
STALL_MINUTES = 90
AWAKE_KEY = "keep_awake"


def keeps_awake(ws: Path) -> bool:
    from .config import load_config

    return bool(load_config(ws).get(AWAKE_KEY, True))


def stall_seconds(ws: Path) -> int:
    from .config import load_config

    return int(60 * float(load_config(ws).get(STALL_KEY) or STALL_MINUTES))
