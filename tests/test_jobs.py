import subprocess
import sys
import time

import pytest

from byosm.app import jobs
from byosm.app.detached import LOG_SUFFIX, alive, job_file
from byosm.app.hub import Hub
from byosm.app.jobs import LOST_LINE, Runner
from byosm.app.store import JobStore
from byosm.events import EVENTS_ENV, emit

SLEEPER = "import sys, time; print('started', flush=True); time.sleep(float(sys.argv[2]))"
EMITTER = "from byosm.events import emit; emit('epoch', epoch=1, loss=0.5); print('@event not json')"
SURVIVOR = "import time; print('started', flush=True); time.sleep(1.5); print('finished')"
EXITER = "import sys; print('bye', flush=True); sys.exit(int(sys.argv[2]))"
GROUP = (
    "import subprocess, sys, time; child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)']); "
    "print(child.pid, flush=True); time.sleep(60)"
)
THREADS = (
    "from concurrent.futures import ThreadPoolExecutor; import time; print('started', flush=True); "
    "pool = ThreadPoolExecutor(4); [pool.submit(time.sleep, 60) for _ in range(8)]; pool.shutdown(wait=True)"
)
CANCEL_WITHIN_SECONDS = 5
POLL_SECONDS = 0.02
TIMEOUT_SECONDS = 20


def wait_for(check):
    end = time.time() + TIMEOUT_SECONDS
    while not check():
        assert time.time() < end, "timed out"
        time.sleep(POLL_SECONDS)


@pytest.fixture
def sleeper(monkeypatch):
    monkeypatch.setattr(jobs, "ENTRY", SLEEPER)


def test_jobs_run_in_parallel_up_to_the_limit(ws, sleeper):
    runner = Runner(ws, Hub(), parallel=2)
    submitted = [runner.submit(["report", "0.5"]) for _ in range(3)]
    most = 0
    while any(j.status in ("queued", "running") for j in submitted):
        most = max(most, sum(j.status == "running" for j in submitted))
        time.sleep(POLL_SECONDS)
    assert most == 2 and all(j.status == "done" for j in submitted)


def test_cancel_stops_the_running_job_only(ws, sleeper):
    runner = Runner(ws, Hub(), parallel=2)
    keep, stop = runner.submit(["report", "1"]), runner.submit(["report", "60"])
    wait_for(lambda: runner.detail(stop.id)["lines"] == ["started"])
    runner.cancel(stop.id)
    wait_for(lambda: stop.status == "cancelled")
    wait_for(lambda: keep.status == "done")
    assert stop.code != 0 and keep.code == 0 and not runner.procs


class Recorder(Hub):
    def __init__(self):
        super().__init__()
        self.seen = []

    def publish(self, event):
        self.seen.append(event)


def test_event_lines_become_metric_events_and_are_stored(ws, monkeypatch):
    monkeypatch.setattr(jobs, "ENTRY", EMITTER)
    hub = Recorder()
    runner = Runner(ws, hub)
    job = runner.submit(["report"])
    wait_for(lambda: job.status == "done")
    metric = {"type": "metric", "job": job.id, "data": {"kind": "epoch", "epoch": 1, "loss": 0.5}}
    seen = [
        {**e, "data": {k: v for k, v in e["data"].items() if k != "at"}} if e["type"] == "metric" else e
        for e in hub.seen
    ]
    assert metric in seen and {"type": "log", "job": job.id, "line": "@event not json"} in seen
    detail = Runner(ws, Hub()).detail(job.id)
    assert [{k: v for k, v in e.items() if k != "at"} for e in detail["events"]] == [metric["data"]] and detail[
        "lines"
    ] == ["@event not json"]


def test_a_job_survives_a_new_runner_without_duplicate_lines(ws, monkeypatch):
    monkeypatch.setattr(jobs, "ENTRY", SURVIVOR)
    first = Runner(ws, Hub())
    job = first.submit(["report"])
    wait_for(lambda: first.detail(job.id)["lines"] == ["started"])
    first.shutdown()
    second = Runner(ws, Hub())
    assert second.jobs[job.id].status == "running"
    wait_for(lambda: second.jobs[job.id].status == "done")
    assert second.detail(job.id)["lines"] == ["started", "finished"] and job.status == "running"


def test_a_dead_job_without_an_exit_file_fails_with_its_log(ws):
    gone = subprocess.Popen([sys.executable, "-c", "pass"])
    gone.wait()
    log = job_file(ws, "lost", LOG_SUFFIX)
    log.parent.mkdir(parents=True)
    log.write_text("partial\n")
    row = {"id": "lost", "args": ["report"], "status": "running", "created": 0.0, "started": 0.0}
    JobStore(ws).save({**row, "ended": None, "code": None, "pid": gone.pid})
    detail = Runner(ws, Hub()).detail("lost")
    assert detail["status"] == "failed" and detail["lines"] == ["partial", LOST_LINE] and not log.exists()


@pytest.mark.parametrize(("code", "status"), [(0, "done"), (3, "failed")])
def test_exit_codes_map_to_done_or_failed(ws, monkeypatch, code, status):
    monkeypatch.setattr(jobs, "ENTRY", EXITER)
    runner = Runner(ws, Hub())
    job = runner.submit(["report", str(code)])
    wait_for(lambda: job.status not in ("queued", "running"))
    assert (job.status, job.code) == (status, code) and runner.detail(job.id)["lines"] == ["bye"]


def test_cancel_kills_the_whole_process_group(ws, monkeypatch):
    monkeypatch.setattr(jobs, "ENTRY", GROUP)
    runner = Runner(ws, Hub())
    job = runner.submit(["report"])
    wait_for(lambda: runner.detail(job.id)["lines"])
    child = int(runner.detail(job.id)["lines"][0])
    runner.cancel(job.id)
    wait_for(lambda: job.status == "cancelled")
    wait_for(lambda: not alive(child))


def test_emit_is_silent_outside_a_job(monkeypatch, capsys):
    monkeypatch.delenv(EVENTS_ENV, raising=False)
    emit("epoch", epoch=1)
    assert capsys.readouterr().out == ""


def test_cancel_stops_a_job_whose_threads_are_still_busy(ws, monkeypatch):
    monkeypatch.setattr(jobs, "ENTRY", THREADS)
    runner = Runner(ws, Hub())
    job = runner.submit(["report"])
    wait_for(lambda: job.status == "running" and job.pid is not None)
    time.sleep(0.5)
    started = time.time()
    runner.cancel(job.id)
    wait_for(lambda: job.status == "cancelled")
    assert time.time() - started < CANCEL_WITHIN_SECONDS and not alive(job.pid)
