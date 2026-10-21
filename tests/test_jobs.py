import time

import pytest

from byosm.app import jobs
from byosm.app.jobs import Hub, Runner

SLEEPER = "import sys, time; print('started', flush=True); time.sleep(float(sys.argv[2]))"
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
    wait_for(lambda: runner.lines(stop.id) == ["started"])
    runner.cancel(stop.id)
    wait_for(lambda: stop.status == "cancelled")
    wait_for(lambda: keep.status == "done")
    assert stop.code != 0 and keep.code == 0 and not runner.procs
