from __future__ import annotations

import json
import os
import queue
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..config import MAX_PARALLEL_JOBS
from ..events import EVENTS_ENV, parse
from .detached import LOG_SUFFIX, Tail, alive, exit_code, forget, job_file, kill_group, spawn
from .hub import Hub
from .store import JOB_FIELDS, JobStore

ALLOWED_COMMANDS = (
    "ls",
    "containerize",
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
    "publish",
    "listen",
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
TERMINAL_COLUMNS = "160"
ENTRY = "from byosm.cli import app; app()"
LOST_LINE = "the job process ended without reporting an exit code"
FOLLOW_SECONDS = 0.1


@dataclass
class Job:
    id: str
    args: list[str]
    status: str = "queued"
    created: float = field(default_factory=time.time)
    started: float | None = None
    ended: float | None = None
    code: int | None = None
    pid: int | None = None
    cancel: bool = False
    llm: dict[str, Any] | None = None

    def public(self) -> dict[str, Any]:
        return {k: getattr(self, k) for k in JOB_FIELDS}


class Runner:
    def __init__(self, ws: Path, hub: Hub, parallel: int = MAX_PARALLEL_JOBS) -> None:
        self.ws, self.hub, self.store = ws, hub, JobStore(ws)
        self.jobs = {row["id"]: Job(**row) for row in self.store.jobs()}
        self.todo: queue.Queue[Job] = queue.Queue()
        self.procs: dict[str, subprocess.Popen] = {}
        self.lock = threading.Lock()
        self.stopped = False
        running = [j for j in self.jobs.values() if j.status == "running"]
        for job in running:
            if not self._settled(job, self._tail(job)):
                self.todo.put(job)
        for job in self.jobs.values():
            if job.status == "queued":
                self.todo.put(job)
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
            pid = job.pid
        if job.status == "queued":
            self._status(job, "cancelled")
        elif pid is not None:
            kill_group(pid)
        return job

    def detail(self, job_id: str) -> dict[str, Any]:
        job = self.jobs[job_id]
        return {**job.public(), "lines": self.store.lines(job.id), "events": self.store.events(job.id)}

    def shutdown(self) -> None:
        self.stopped = True

    def _status(self, job: Job, status: str) -> None:
        job.status = status
        self.store.save(job.public())
        self.hub.publish({"type": "job", "job": job.public()})

    def _line(self, job: Job, line: str) -> None:
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
        while not self.stopped:
            job = self.todo.get()
            if self.stopped:
                return
            if job.status == "queued":
                self._run(job)
            elif job.status == "running":
                self._follow(job)

    def _env(self, job: Job) -> dict[str, str]:
        from ..config import LLM_OVERRIDE_ENV, env_file

        return {
            **os.environ,
            **env_file(self.ws),
            "PYTHONUNBUFFERED": "1",
            "COLUMNS": TERMINAL_COLUMNS,
            "NO_COLOR": "1",
            EVENTS_ENV: "1",
            **({LLM_OVERRIDE_ENV: json.dumps(job.llm)} if job.llm else {}),
        }

    def _run(self, job: Job) -> None:
        forget(self.ws, job.id)
        proc = spawn(self.ws, job.id, ENTRY, job.args, self._env(job))
        with self.lock:
            self.procs[job.id] = proc
            job.pid, job.started = proc.pid, time.time()
            stop = job.cancel
        if stop:
            kill_group(proc.pid)
        self._status(job, "running")
        self._follow(job)

    def _tail(self, job: Job) -> Tail:
        return Tail(job_file(self.ws, job.id, LOG_SUFFIX), self.store.seen(job.id))

    def _follow(self, job: Job) -> None:
        tail = self._tail(job)
        while not self.stopped and not self._settled(job, tail):
            time.sleep(FOLLOW_SECONDS)

    def _running(self, job: Job) -> bool:
        proc = self.procs.get(job.id)
        if proc is not None:
            return proc.poll() is None
        return job.pid is not None and alive(job.pid)

    def _settled(self, job: Job, tail: Tail) -> bool:
        for line in tail.read():
            self._output(job, line)
        code = exit_code(self.ws, job.id)
        if code is None and self._running(job):
            return False
        code = exit_code(self.ws, job.id) if code is None else code
        for line in tail.read(final=True):
            self._output(job, line)
        if code is None:
            self._line(job, LOST_LINE)
        job.code, job.ended = code, time.time()
        with self.lock:
            proc = self.procs.pop(job.id, None)
        if proc is not None:
            proc.wait()
        self._status(job, "cancelled" if job.cancel else "done" if code == 0 else "failed")
        forget(self.ws, job.id)
        return True
