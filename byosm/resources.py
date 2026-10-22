from __future__ import annotations

import os

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
