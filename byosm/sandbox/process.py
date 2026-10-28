from __future__ import annotations

import contextlib
import resource
import sys
from typing import Any

import psutil

from ..config import PROCESS
from ..resources import GB

HOUR_S = 3600
UNCAPPED_DEVICE = "cuda"


def _cap(proc: psutil.Process, which: int, value: int) -> None:
    hard = proc.rlimit(which)[1]
    value = value if hard == resource.RLIM_INFINITY else min(value, hard)
    proc.rlimit(which, (value, value))


def restrict(pid: int, limits: dict[str, Any], device: str) -> dict[str, Any]:
    applied: dict[str, Any] = {"mode": PROCESS}
    linux = sys.platform.startswith("linux")
    with contextlib.suppress(psutil.Error):
        proc = psutil.Process(pid)
        nice = max(limits["nice"], proc.nice())
        proc.nice(nice)
        applied["nice"] = nice
        if linux and device != UNCAPPED_DEVICE:
            _cap(proc, resource.RLIMIT_AS, int(limits["memory_gb"] * GB))
            applied["memory_gb"] = limits["memory_gb"]
        if linux and limits["cpu_hours"]:
            _cap(proc, resource.RLIMIT_CPU, int(limits["cpu_hours"] * HOUR_S))
            applied["cpu_hours"] = limits["cpu_hours"]
    return applied
