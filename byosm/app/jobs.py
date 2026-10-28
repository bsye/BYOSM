from __future__ import annotations

import contextlib
import queue
import subprocess
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from ..events import TARGET, parse
from ..resources import can_start_job, keeps_awake, max_jobs, stall_seconds
from ..sandbox.launch import start, worker_cpu
from .allowed import ALLOWED_COMMANDS, PERSON_ONLY
from .detached import LOG_SUFFIX, Tail, ending, exit_code, forget, job_file, keep_awake, kill_group, owned
from .hub import Hub
from .job import Job
from .store import JobStore
from .watchdog import Watch

ADMIT_POLL_S = 2
ENTRY = "from byosm.cli import app; app()"
LOST_LINE = "the job process ended without reporting an exit code"
FOLLOW_SECONDS = 0.1
FAILED = "failed"


class Runner:
    def __init__(self, ws: Path, hub: Hub, parallel: int | None = None) -> None:
        self.ws, self.hub, self.store = ws, hub, JobStore(ws)
        targets, boxes = self.store.targets(), self.store.isolations()
        self.jobs = {
            r["id"]: Job(**r, target=targets.get(r["id"]), isolation=boxes.get(r["id"])) for r in self.store.jobs()
        }
        self.todo: queue.Queue[Job] = queue.Queue()
        self.procs: dict[str, subprocess.Popen] = {}
        self.lock = threading.Lock()
        self.stopped = False
        live = [j for j in self.jobs.values() if j.status == "running" and not self._settled(j, self._tail(j))]
        for job in live + [j for j in self.jobs.values() if j.status == "queued"]:
            self.todo.put(job)
        self.cap = parallel or max_jobs()
        for _ in range(max(1, self.cap)):
            threading.Thread(target=self._work, daemon=True).start()

    def submit(self, args: list[str], llm: dict[str, Any] | None = None) -> Job:
        if not args or args[0] not in ALLOWED_COMMANDS:
            raise ValueError(f"command must be one of {', '.join(ALLOWED_COMMANDS)}")
        if tuple(args[: len(PERSON_ONLY[0])]) in PERSON_ONLY:
            raise ValueError("approving lab code is a person's decision: use the Lab tab, not a job")
        job = Job(id=uuid.uuid4().hex[:10], args=[str(a) for a in args], llm=llm)
        with self.lock:
            self.jobs[job.id] = job
        self._status(job, "queued")
        self.todo.put(job)
        return job

    def cancel(self, job_id: str) -> Job:
        job = self.jobs[job_id]
        with self.lock:
            job.cancel = True
            queued = job.pid is None and job.status == "queued"
        if queued:
            self._status(job, "cancelled")
        else:
            self._kill(job)
        return job

    def snapshot(self) -> list[Job]:
        with self.lock:
            return list(self.jobs.values())

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
        if data.get("kind") == TARGET:
            job.target = data.get("name")
            self.hub.publish({"type": "job", "job": job.public()})
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
            try:
                if job.status == "queued":
                    self._admit()
                    if not job.cancel:
                        self._run(job)
                elif job.status == "running":
                    self._follow(job)
            except Exception as error:
                self._fail(job, error)

    def _fail(self, job: Job, error: Exception) -> None:
        with self.lock:
            self.procs.pop(job.id, None)
            job.ended = time.time()
        with contextlib.suppress(Exception):
            self._kill(job)
        with contextlib.suppress(Exception):
            self._status(job, FAILED)
            self._line(job, f"stopped: the job runner hit an error: {error!r}")

    def _kill(self, job: Job) -> None:
        if job.pid is not None and self._running(job):
            kill_group(job.pid)

    def _admit(self) -> None:
        while not self.stopped:
            running = sum(j.status == "running" for j in self.snapshot())
            if running < self.cap and can_start_job(running):
                return
            time.sleep(ADMIT_POLL_S)

    def _run(self, job: Job) -> None:
        forget(self.ws, job.id)
        proc = start(self.ws, job, ENTRY)
        self.store.isolate(job.id, job.isolation)
        if keeps_awake(self.ws):
            keep_awake(proc.pid)
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
        tail, watch, killed = self._tail(job), Watch(worker_cpu(job), stall_seconds(self.ws)), False
        while not self.stopped and not self._settled(job, tail, watch.heard):
            if not killed and watch.stuck():
                minutes = watch.limit // 60
                self._line(job, f"stopped: no output and no CPU work for {minutes} minutes, the job looked stuck")
                self._kill(job)
                killed = job.stalled = True
            time.sleep(FOLLOW_SECONDS)

    def _running(self, job: Job) -> bool:
        proc = self.procs.get(job.id)
        if proc is not None:
            return proc.poll() is None
        return job.pid is not None and owned(job.pid, job.started or 0)

    def _settled(self, job: Job, tail: Tail, heard: list[float] | None = None) -> bool:
        fresh = tail.read()
        if fresh and heard is not None:
            heard[0] = time.time()
        for line in fresh:
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
        self._status(job, ending(code, job.cancel, job.stalled))
        forget(self.ws, job.id)
        return True
