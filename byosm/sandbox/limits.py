from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import psutil

from ..config import job_settings
from ..resources import GB

JOB_MEMORY_SHARE = 0.75
RESERVED_CORES = 1


def job_limits(ws: Path) -> dict[str, Any]:
    settings = job_settings(ws)
    cores = os.cpu_count() or 1
    memory = settings["memory_gb"] or psutil.virtual_memory().total * JOB_MEMORY_SHARE / GB
    return {
        "cpus": float(settings["cpus"] or max(1, cores - RESERVED_CORES)),
        "memory_gb": round(float(memory), 1),
        "gpus": str(settings["gpus"]),
        "nice": int(settings["nice"]),
        "cpu_hours": settings["cpu_hours"],
    }
