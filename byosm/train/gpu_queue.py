from __future__ import annotations

import os
import time
from pathlib import Path

QUEUE = "queue"
TICKET = ".ticket"


def join(base: Path, need: float) -> Path:
    mine = base / QUEUE / f"{os.getpid()}{TICKET}"
    mine.parent.mkdir(parents=True, exist_ok=True)
    if not mine.exists():
        mine.write_text(f"{time.time()} {need}")
    return mine


def _read(ticket: Path) -> tuple[float, float]:
    try:
        since, _, need = ticket.read_text().partition(" ")
    except FileNotFoundError:
        return 0.0, 0.0
    return float(since or 0), float(need or 0)


def first(mine: Path, room: float) -> bool:
    import psutil

    since = _read(mine)[0]
    for ticket in mine.parent.glob(f"*{TICKET}"):
        if not psutil.pid_exists(int(ticket.stem)):
            ticket.unlink(missing_ok=True)
        elif ticket != mine and (ticket_since := _read(ticket))[0] < since and ticket_since[1] <= room:
            return False
    return True
