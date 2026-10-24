from __future__ import annotations

import time

from .detached import cpu_seconds

SLEEP_GAP_S = 30
BUSY_CPU_S = 60


class Watch:
    def __init__(self, pid: int | None, limit: int) -> None:
        self.pid, self.limit = pid, limit
        self.heard, self.last = [time.time()], time.time()
        self.spent = cpu_seconds(pid) if pid else 0.0

    def stuck(self) -> bool:
        now = time.time()
        if now - self.last > SLEEP_GAP_S:
            self.heard[0] += now - self.last
        self.last = now
        if not self.pid or now - self.heard[0] <= self.limit:
            return False
        cpu = cpu_seconds(self.pid)
        busy, self.spent = cpu - self.spent >= BUSY_CPU_S, cpu
        if busy:
            self.heard[0] = now
        return not busy
