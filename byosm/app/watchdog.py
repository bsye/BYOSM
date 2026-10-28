from __future__ import annotations

import time
from collections.abc import Callable

SLEEP_GAP_S = 30
BUSY_CPU_S = 60


class Watch:
    def __init__(self, cpu: Callable[[], float] | None, limit: int) -> None:
        self.cpu, self.limit = cpu, limit
        self.heard, self.last = [time.time()], time.time()
        self.spent = cpu() if cpu else 0.0

    def stuck(self) -> bool:
        now = time.time()
        if now - self.last > SLEEP_GAP_S:
            self.heard[0] += now - self.last
        self.last = now
        if not self.cpu or now - self.heard[0] <= self.limit:
            return False
        cpu = self.cpu()
        busy, self.spent = cpu - self.spent >= BUSY_CPU_S, cpu
        if busy:
            self.heard[0] = now
        return not busy
