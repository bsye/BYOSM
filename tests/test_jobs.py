import time

import pytest

from byosm.app import jobs
from byosm.app.hub import Hub
from byosm.app.jobs import Runner
from byosm.events import EVENTS_ENV, emit

SLEEPER = "import sys, time; print('started', flush=True); time.sleep(float(sys.argv[2]))"
EMITTER = "from byosm.events import emit; emit('epoch', epoch=1, loss=0.5); print('@event not json')"
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
    assert metric in hub.seen and {"type": "log", "job": job.id, "line": "@event not json"} in hub.seen
    detail = Runner(ws, Hub()).detail(job.id)
    assert detail["events"] == [metric["data"]] and detail["lines"] == ["@event not json"]


def test_emit_is_silent_outside_a_job(monkeypatch, capsys):
    monkeypatch.delenv(EVENTS_ENV, raising=False)
    emit("epoch", epoch=1)
    assert capsys.readouterr().out == ""
