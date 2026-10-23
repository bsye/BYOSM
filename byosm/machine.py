from __future__ import annotations

import os
import platform
import shutil
import statistics
import subprocess
from pathlib import Path
from typing import Any

from sqlalchemy import select

from .db import Experiment, session
from .train.gpu import busy_slots, slots

ADAM_BYTES_PER_PARAM = 16
FROZEN_BYTES_PER_PARAM = 4
ACTIVATION_OVERHEAD = 1.5
UNIFIED_MEMORY_SHARE = 0.75
BYOSM_PROCESS = "byosm"
TYPICAL_ROWS = 10_000
CPU_SAMPLE_S = 0.2
LOW_MEMORY_GB = 8
TYPICAL_EPOCHS = 4


def _cpu_name() -> str:
    if platform.system() == "Darwin":
        out = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True)
        return out.stdout.strip() or platform.processor()
    return platform.processor() or platform.machine()


def _accelerator(ram: int) -> dict[str, Any]:
    import torch

    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        return {"kind": "cuda", "name": props.name, "memory": props.total_memory}
    if torch.backends.mps.is_available():
        return {"kind": "mps", "name": "Apple GPU (unified memory)", "memory": int(ram * UNIFIED_MEMORY_SHARE)}
    return {"kind": "cpu", "name": "CPU only", "memory": int(ram * UNIFIED_MEMORY_SHARE)}


def _battery() -> dict[str, Any] | None:
    import psutil

    b = psutil.sensors_battery()
    return {"percent": round(b.percent), "plugged": bool(b.power_plugged)} if b else None


def _byosm_memory() -> int:
    import psutil

    total = 0
    for proc in psutil.process_iter(["cmdline", "memory_info"]):
        cmd = " ".join(proc.info.get("cmdline") or [])
        if BYOSM_PROCESS in cmd and proc.info.get("memory_info"):
            total += proc.info["memory_info"].rss
    return total


def training_bytes(params: int, frozen: bool = False) -> int:
    per = FROZEN_BYTES_PER_PARAM if frozen else ADAM_BYTES_PER_PARAM
    return int(params * per * ACTIVATION_OVERHEAD)


def _speed(meta: dict) -> float | None:
    seconds, rows, epochs = meta.get("train_seconds"), meta.get("n_train"), len(meta.get("history") or [])
    return rows * epochs / seconds if seconds and rows and epochs else None


def backbones(ws: Path) -> list[dict[str, Any]]:
    with session(ws) as s:
        found = s.execute(select(Experiment.spec, Experiment.meta)).all()
    runs: dict[str, list[tuple[int, float]]] = {}
    for spec, meta in found:
        speed, params = _speed(meta or {}), (meta or {}).get("params")
        if speed and params and spec:
            runs.setdefault(spec.get("backbone") or "", []).append((params, speed))
    out = []
    for name, rows in runs.items():
        params = max(p for p, _ in rows)
        speed = statistics.median(v for _, v in rows)
        out.append(
            {
                "backbone": name,
                "params": params,
                "rows_per_second": round(speed, 1),
                "runs": len(rows),
                "memory": training_bytes(params),
                "typical_seconds": round(TYPICAL_ROWS * TYPICAL_EPOCHS / speed),
            }
        )
    return sorted(out, key=lambda b: b["params"])


def memory() -> dict[str, Any]:
    import psutil

    from .train.gpu import GB_PER_TRAINING

    mem = psutil.virtual_memory()
    return {
        "available": mem.available,
        "wired": getattr(mem, "wired", 0),
        "total": mem.total,
        "low_gb": LOW_MEMORY_GB,
        "training_gb": GB_PER_TRAINING,
    }


def machine(ws: Path) -> dict[str, Any]:
    import psutil

    mem = psutil.virtual_memory()
    accel = _accelerator(mem.total)
    disk = shutil.disk_usage(ws)
    cores = os.cpu_count() or 1
    return {
        "cpu": {
            "name": _cpu_name(),
            "cores": cores,
            "load": round(os.getloadavg()[0] / cores, 2),
            "percent": psutil.cpu_percent(interval=CPU_SAMPLE_S),
        },
        "memory": {
            "total": mem.total,
            "available": mem.available,
            "byosm": _byosm_memory(),
            "wired": getattr(mem, "wired", 0),
        },
        "accelerator": accel,
        "disk": {"total": disk.total, "free": disk.free},
        "battery": _battery(),
        "largest": {
            "full": accel["memory"] // training_bytes(1),
            "frozen": accel["memory"] // training_bytes(1, frozen=True),
        },
        "backbones": backbones(ws),
        "gpu_slots": {"total": slots(), "busy": busy_slots(ws)},
    }
