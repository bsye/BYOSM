from __future__ import annotations

import asyncio
import os
import queue
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ALLOWED_COMMANDS = (
    "new",
    "build",
    "data",
    "eval",
    "improve",
    "auto",
    "distill",
    "export",
    "report",
    "request-data",
    "check-data",
    "promote",
    "exp",
    "suite",
    "baseline",
    "system",
)
BACKLOG_LINES = 5000
TERMINAL_COLUMNS = "160"
ENTRY = "from byosm.cli import app; app()"


@dataclass
class Job:
    id: str
    args: list[str]
    status: str = "queued"
    created: float = field(default_factory=time.time)
    started: float | None = None
    ended: float | None = None
    code: int | None = None
    lines: list[str] = field(default_factory=list)
    cancel: bool = False

    def public(self) -> dict[str, Any]:
        keys = ("id", "args", "status", "created", "started", "ended", "code")
        return {k: getattr(self, k) for k in keys}


class Hub:
    def __init__(self) -> None:
        self.subs: set[tuple[asyncio.AbstractEventLoop, asyncio.Queue]] = set()
        self.lock = threading.Lock()

    def publish(self, event: dict[str, Any]) -> None:
        with self.lock:
            subs = list(self.subs)
        for loop, q in subs:
            loop.call_soon_threadsafe(q.put_nowait, event)

    def subscribe(self) -> tuple[asyncio.AbstractEventLoop, asyncio.Queue]:
        sub = (asyncio.get_running_loop(), asyncio.Queue())
        with self.lock:
            self.subs.add(sub)
        return sub

    def unsubscribe(self, sub: tuple[asyncio.AbstractEventLoop, asyncio.Queue]) -> None:
        with self.lock:
            self.subs.discard(sub)


class Runner:
    def __init__(self, ws: Path, hub: Hub) -> None:
        self.ws, self.hub = ws, hub
        self.jobs: dict[str, Job] = {}
        self.todo: queue.Queue[Job] = queue.Queue()
        self.proc: subprocess.Popen | None = None
        threading.Thread(target=self._work, daemon=True).start()

    def submit(self, args: list[str]) -> Job:
        if not args or args[0] not in ALLOWED_COMMANDS:
            raise ValueError(f"command must be one of {', '.join(ALLOWED_COMMANDS)}")
        job = Job(id=uuid.uuid4().hex[:10], args=[str(a) for a in args])
        self.jobs[job.id] = job
        self._status(job, "queued")
        self.todo.put(job)
        return job

    def cancel(self, job_id: str) -> Job:
        job = self.jobs[job_id]
        job.cancel = True
        if job.status == "queued":
            self._status(job, "cancelled")
        elif job.status == "running" and self.proc is not None:
            self.proc.terminate()
        return job

    def shutdown(self) -> None:
        if self.proc is not None and self.proc.poll() is None:
            self.proc.terminate()

    def _status(self, job: Job, status: str) -> None:
        job.status = status
        self.hub.publish({"type": "job", "job": job.public()})

    def _line(self, job: Job, line: str) -> None:
        job.lines = (job.lines + [line])[-BACKLOG_LINES:]
        self.hub.publish({"type": "log", "job": job.id, "line": line})

    def _work(self) -> None:
        while True:
            job = self.todo.get()
            if job.status == "queued":
                self._run(job)

    def _run(self, job: Job) -> None:
        from ..config import env_file

        env = {**os.environ, **env_file(self.ws), "PYTHONUNBUFFERED": "1", "COLUMNS": TERMINAL_COLUMNS, "NO_COLOR": "1"}
        job.started = time.time()
        self._status(job, "running")
        self.proc = subprocess.Popen(
            [sys.executable, "-c", ENTRY, *job.args],
            cwd=self.ws,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        for raw in self.proc.stdout or []:
            for line in raw.rstrip("\n").split("\r"):
                if line.strip():
                    self._line(job, line)
        job.code, job.ended = self.proc.wait(), time.time()
        self.proc = None
        self._status(job, "cancelled" if job.cancel else "done" if job.code == 0 else "failed")
