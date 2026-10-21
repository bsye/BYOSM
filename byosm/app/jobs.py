from __future__ import annotations

import json
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

from ..config import MAX_PARALLEL_JOBS
from ..events import EVENTS_ENV, parse
from .hub import Hub
from .store import JOB_FIELDS, JobStore

ALLOWED_COMMANDS = (
    "ls",
    "show",
    "board",
    "new",
    "build",
    "data",
    "eval",
    "overfit",
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
    "testset",
    "system",
)
BACKLOG_LINES = 5000
TERMINAL_COLUMNS = "160"
ENTRY = "from byosm.cli import app; app()"
UNFINISHED = ("queued", "running")
STOPPED_LINE = "server stopped while this job ran"


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
    llm: dict[str, Any] | None = None

    def public(self) -> dict[str, Any]:
        return {k: getattr(self, k) for k in JOB_FIELDS}


class Runner:
    def __init__(self, ws: Path, hub: Hub, parallel: int = MAX_PARALLEL_JOBS) -> None:
        self.ws, self.hub, self.store = ws, hub, JobStore(ws)
        self.jobs = {j.id: j for j in map(self._restored, self.store.jobs())}
        self.todo: queue.Queue[Job] = queue.Queue()
        self.procs: dict[str, subprocess.Popen] = {}
        self.lock = threading.Lock()
        for _ in range(max(1, parallel)):
            threading.Thread(target=self._work, daemon=True).start()

    def submit(self, args: list[str], llm: dict[str, Any] | None = None) -> Job:
        if not args or args[0] not in ALLOWED_COMMANDS:
            raise ValueError(f"command must be one of {', '.join(ALLOWED_COMMANDS)}")
        job = Job(id=uuid.uuid4().hex[:10], args=[str(a) for a in args], llm=llm)
        self.jobs[job.id] = job
        self._status(job, "queued")
        self.todo.put(job)
        return job

    def cancel(self, job_id: str) -> Job:
        job = self.jobs[job_id]
        with self.lock:
            job.cancel = True
            proc = self.procs.get(job.id)
        if job.status == "queued":
            self._status(job, "cancelled")
        elif proc is not None:
            proc.terminate()
        return job

    def detail(self, job_id: str) -> dict[str, Any]:
        job = self.jobs[job_id]
        return {**job.public(), "lines": self.store.lines(job.id), "events": self.store.events(job.id)}

    def _restored(self, row: dict[str, Any]) -> Job:
        job = Job(**row)
        if job.status in UNFINISHED:
            self.store.line(job.id, STOPPED_LINE)
            job.status, job.ended = "failed", job.ended or time.time()
            self.store.save(job.public())
        return job

    def shutdown(self) -> None:
        with self.lock:
            procs = list(self.procs.values())
        for proc in procs:
            proc.terminate()

    def _status(self, job: Job, status: str) -> None:
        job.status = status
        self.store.save(job.public())
        self.hub.publish({"type": "job", "job": job.public()})

    def _line(self, job: Job, line: str) -> None:
        job.lines = (job.lines + [line])[-BACKLOG_LINES:]
        self.store.line(job.id, line)
        self.hub.publish({"type": "log", "job": job.id, "line": line})

    def _metric(self, job: Job, data: dict[str, Any]) -> None:
        self.store.event(job.id, data)
        self.hub.publish({"type": "metric", "job": job.id, "data": data})

    def _output(self, job: Job, line: str) -> None:
        data = parse(line)
        if data is None:
            self._line(job, line)
        else:
            self._metric(job, data)

    def _work(self) -> None:
        while True:
            job = self.todo.get()
            if job.status == "queued":
                self._run(job)

    def _run(self, job: Job) -> None:
        from ..config import LLM_OVERRIDE_ENV, env_file

        env = {
            **os.environ,
            **env_file(self.ws),
            "PYTHONUNBUFFERED": "1",
            "COLUMNS": TERMINAL_COLUMNS,
            "NO_COLOR": "1",
            EVENTS_ENV: "1",
            **({LLM_OVERRIDE_ENV: json.dumps(job.llm)} if job.llm else {}),
        }
        job.started = time.time()
        self._status(job, "running")
        proc = subprocess.Popen(
            [sys.executable, "-c", ENTRY, *job.args],
            cwd=self.ws,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        with self.lock:
            self.procs[job.id] = proc
            stop = job.cancel
        if stop:
            proc.terminate()
        for raw in proc.stdout or []:
            for line in raw.rstrip("\n").split("\r"):
                if line.strip():
                    self._output(job, line)
        job.code, job.ended = proc.wait(), time.time()
        with self.lock:
            del self.procs[job.id]
        self._status(job, "cancelled" if job.cancel else "done" if job.code == 0 else "failed")
