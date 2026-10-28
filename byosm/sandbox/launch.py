from __future__ import annotations

import json
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ..app.detached import EXIT_SUFFIX, INNER_SUFFIX, cpu_seconds, job_file, note, spawn, wrapped
from ..app.job import Job, job_env
from ..config import CONTAINER, device, job_settings
from .allow import allow_list, connections
from .choose import blocker
from .command import gpu_flags, run_args
from .containers import JOB_PREFIX, container_pid
from .image import training_image
from .limits import job_limits
from .process import restrict

SUPERVISE_ENTRY = "from byosm.sandbox.supervise import main; main()"
FALLBACK = "fallback"
FALLBACK_LINE = "isolation: running as a process, not in a container, because {}"


def start(ws: Path, job: Job, entry: str) -> subprocess.Popen:
    limits, dev = job_limits(ws), device()
    wanted = job_settings(ws)["isolation"] == CONTAINER
    why = blocker(ws, connections(ws, job.llm), dev) if wanted else None
    if wanted and why is None:
        return _contained(ws, job, entry, limits, dev)
    if why:
        note(ws, job.id, FALLBACK_LINE.format(why))
    proc = spawn(ws, job.id, wrapped(job_file(ws, job.id, EXIT_SUFFIX), entry, job.args), job_env(ws, job))
    job.isolation = {**restrict(proc.pid, limits, dev), **({FALLBACK: why} if why else {})}
    return proc


def _contained(ws: Path, job: Job, entry: str, limits: dict[str, Any], dev: str) -> subprocess.Popen:
    built, name = training_image(dev == "cuda"), JOB_PREFIX + job.id
    gpus = gpu_flags(limits["gpus"]) if dev == "cuda" else []
    rules, inner = allow_list(ws, connections(ws, job.llm)), job_file(ws, job.id, INNER_SUFFIX)
    shown = {"cpus": limits["cpus"], "memory_gb": limits["memory_gb"], "gpus": limits["gpus"] if gpus else None}
    run = run_args(ws, job, entry, built["image"], limits, gpus, inner)
    plan = {**built, **shown, "job": job.id, "container": name, "network": rules, "inner": str(inner), "run": run}
    command = wrapped(job_file(ws, job.id, EXIT_SUFFIX), SUPERVISE_ENTRY, [json.dumps(plan)])
    job.isolation = {"mode": CONTAINER, "container": name, **shown, "network": rules}
    return spawn(ws, job.id, command, job_env(ws, job))


def _container_cpu(name: str) -> float:
    pid = container_pid(name)
    return cpu_seconds(pid) if pid else 0.0


def worker_cpu(job: Job) -> Callable[[], float] | None:
    name, pid = (job.isolation or {}).get("container"), job.pid
    if name:
        return lambda: _container_cpu(name)
    return (lambda: cpu_seconds(pid)) if pid else None
