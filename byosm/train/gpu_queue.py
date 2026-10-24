from __future__ import annotations

import os
import time
from pathlib import Path

QUEUE = "queue"
TICKET = ".ticket"


def join(base: Path) -> Path:
    mine = base / QUEUE / f"{os.getpid()}{TICKET}"
    mine.parent.mkdir(parents=True, exist_ok=True)
    if not mine.exists():
        mine.write_text(f"{time.time()}")
    return mine


def first(mine: Path) -> bool:
    import psutil

    waiting = []
    for ticket in mine.parent.glob(f"*{TICKET}"):
        if not psutil.pid_exists(int(ticket.stem)):
            ticket.unlink(missing_ok=True)
            continue
        waiting.append((float(ticket.read_text() or 0), ticket.name))
    return not waiting or min(waiting)[1] == mine.name
