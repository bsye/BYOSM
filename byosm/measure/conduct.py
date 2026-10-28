from __future__ import annotations

import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .plan import GPU, steps
from .run import ACTIVE, DONE, FAILED, OPEN, queue, save_run, saved_run

TICK_S = 5.0
MAX_JOBS = 3
DONE_JOB = "done"
LOST = "lost"
MEASURE_EVENT = {"type": "measure"}


def gpu_turn(cells: list[dict[str, Any]]) -> str | None:
    waiting = [c for c in cells if c["lane"] == GPU and c["state"] == OPEN]
    busy = next((c["specialist"] for c in waiting if c["job"]), None)
    return busy or next((c["specialist"] for c in waiting), None)


def _end(c: dict[str, Any], state: str, note: str | None = None) -> bool:
    c.update(state=state, note=note)
    return True


class Tick:
    def __init__(self, ws: Path, runner: Any) -> None:
        self.ws, self.runner = ws, runner
        self.jobs = {j.id: j for j in runner.snapshot()}
        self.found: dict[tuple[str, str], list[dict[str, Any]]] = {}

    def step(self, c: dict[str, Any]) -> dict[str, Any] | None:
        who = (c["specialist"], c["kind"])
        if who not in self.found:
            self.found[who] = steps(self.ws, *who)
        return next((s for s in self.found[who] if s["id"] == c["measurement"]), None)

    def start(self, args: list[str]) -> Any:
        same = next((j for j in self.jobs.values() if j.args == args and j.status in ACTIVE), None)
        job = same or self.runner.submit(args)
        self.jobs[job.id] = job
        return job

    def advance(self, c: dict[str, Any], cells: list[dict[str, Any]]) -> bool:
        job = self.jobs.get(c["job"]) if c["job"] else None
        if job is not None and job.status in ACTIVE:
            return False
        if c["job"] and (job is None or job.status != DONE_JOB):
            return _end(c, FAILED, f"job {job.status if job else LOST}")
        if c["lane"] == GPU and gpu_turn(cells) != c["specialist"]:
            return False
        step = self.step(c)
        if step is None:
            return _end(c, DONE)
        if len(c["jobs"]) >= MAX_JOBS:
            return _end(c, FAILED, f"still missing after {len(c['jobs'])} jobs")
        job = self.start(step["args"])
        c.update(args=step["args"], why=step["why"], model=step["model"], job=job.id, jobs=[*c["jobs"], job.id])
        return True


def tick(ws: Path, runner: Any) -> bool:
    run = saved_run(ws)
    t = Tick(ws, runner)
    moved = [t.advance(c, run["cells"]) for c in run["cells"] if c["state"] == OPEN]
    return any(moved) and save_run(ws, run)


class Conductor:
    def __init__(self, ws: Path, runner: Any, publish: Callable[[dict[str, Any]], None]) -> None:
        self.ws, self.runner, self.publish = ws, runner, publish
        self.wake, self.lock = threading.Event(), threading.Lock()
        self.stopped, self.error = False, None
        threading.Thread(target=self._loop, daemon=True).start()

    def queue(self, names: list[str] | None) -> list[dict[str, Any]]:
        with self.lock:
            fresh = queue(self.ws, names)
        self.wake.set()
        return fresh

    def stop(self) -> None:
        self.stopped = True
        self.wake.set()

    def _loop(self) -> None:
        while not self.stopped:
            self.wake.wait(TICK_S)
            self.wake.clear()
            if not self.stopped:
                self._tick()

    def _tick(self) -> None:
        before = self.error
        try:
            with self.lock:
                moved = tick(self.ws, self.runner)
            self.error = None
        except Exception as e:
            moved, self.error = False, f"{type(e).__name__}: {e}"
        if moved or self.error != before:
            self.publish(MEASURE_EVENT)
