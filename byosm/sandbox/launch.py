from __future__ import annotations

import subprocess
from collections.abc import Callable
from pathlib import Path

from ..app.detached import EXIT_SUFFIX, cpu_seconds, job_file, spawn, wrapped
from ..app.job import Job, job_env
from ..config import device
from .limits import job_limits
from .process import restrict


def start(ws: Path, job: Job, entry: str) -> subprocess.Popen:
    limits, dev = job_limits(ws), device()
    proc = spawn(ws, job.id, wrapped(job_file(ws, job.id, EXIT_SUFFIX), entry, job.args), job_env(ws, job))
    job.isolation = restrict(proc.pid, limits, dev)
    return proc


def worker_cpu(job: Job) -> Callable[[], float] | None:
    pid = job.pid
    return (lambda: cpu_seconds(pid)) if pid else None
