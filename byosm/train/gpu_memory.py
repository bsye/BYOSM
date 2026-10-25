from __future__ import annotations

import json
import math
from pathlib import Path

GB = 1024**3
GB_PER_TRAINING = 16

NEEDS = "needs.json"
OOM_HEADROOM = 1.25
PEAK = [0.0]


def sample_peak() -> None:
    import torch

    if torch.backends.mps.is_available():
        PEAK[0] = max(PEAK[0], torch.mps.driver_allocated_memory() / GB)


def need_gb(base: Path, key: str | None) -> float:
    path = base / NEEDS
    return float(json.loads(path.read_text()).get(key, 0.0)) if key and path.exists() else 0.0


def remember(base: Path, key: str | None, ran_out_at: float | None) -> None:
    if not key:
        return
    sample_peak()
    path = base / NEEDS
    needs = json.loads(path.read_text()) if path.exists() else {}
    seen = max(PEAK[0], (ran_out_at or 0.0) * OOM_HEADROOM)
    needs[key] = math.ceil(max(needs.get(key, 0.0), seen) * 10) / 10
    path.write_text(json.dumps(needs))


def total_gb() -> float:
    import psutil

    return psutil.virtual_memory().total / GB


def free_gb() -> float:
    import psutil

    return psutil.virtual_memory().available / GB


def needed_gb() -> float:
    import psutil

    from ..machine import UNIFIED_MEMORY_SHARE

    return min(GB_PER_TRAINING, psutil.virtual_memory().total * UNIFIED_MEMORY_SHARE / GB)


def release_cache() -> None:
    import torch

    if torch.backends.mps.is_available():
        sample_peak()
        torch.mps.empty_cache()
    elif torch.cuda.is_available():
        torch.cuda.empty_cache()
