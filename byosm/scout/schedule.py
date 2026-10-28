from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from .proposals import due

CHECK_EVERY_S = 3600
COMMAND = "scout"


def tick(ws: Path, runner: Any, cfg: dict, now: float) -> bool:
    busy = any(j.args[:1] == [COMMAND] and j.status in ("queued", "running") for j in runner.snapshot())
    if busy or not due(ws, cfg, now):
        return False
    runner.submit([COMMAND])
    return True


def run_schedule(ws: Path, runner: Any) -> None:
    from ..config import load_config

    while True:
        time.sleep(CHECK_EVERY_S)
        try:
            tick(ws, runner, load_config(ws), time.time())
        except Exception as e:
            print(f"scout schedule: {type(e).__name__}: {e}")
