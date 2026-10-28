import subprocess
import sys
import time

from byosm.app import jobs
from byosm.app.hub import Hub
from byosm.app.job import Job
from byosm.app.jobs import FAILED, Runner
from byosm.app.store import JobStore

QUICK = "print('ok', flush=True)"
NAP = [sys.executable, "-c", "import time; time.sleep(30)"]
POLL_SECONDS = 0.02
TIMEOUT_SECONDS = 20
ADMIT_WAIT_S = 0.3


def wait_for(check):
    end = time.time() + TIMEOUT_SECONDS
    while not check():
        assert time.time() < end, "timed out"
        time.sleep(POLL_SECONDS)


def test_a_worker_survives_an_error_and_runs_the_next_job(ws, monkeypatch):
    monkeypatch.setattr(jobs, "ENTRY", QUICK)
    real = jobs.start
    calls = []

    def flaky(*args):
        calls.append(args)
        if len(calls) == 1:
            raise OSError("no more processes")
        return real(*args)

    monkeypatch.setattr(jobs, "start", flaky)
    runner = Runner(ws, Hub(), parallel=1)
    broken, fine = runner.submit(["report"]), runner.submit(["report"])
    wait_for(lambda: fine.status == "done")
    assert broken.status == FAILED and "no more processes" in runner.detail(broken.id)["lines"][-1]


def test_snapshot_is_a_copy_safe_to_iterate_while_jobs_are_added(ws):
    runner = Runner(ws, Hub(), parallel=1)
    seen = runner.snapshot()
    runner.jobs["extra"] = Job(id="extra", args=["report"], status="done")
    assert seen == [] and [j.id for j in runner.snapshot()] == ["extra"]


def test_a_recovered_job_never_signals_a_process_that_reused_its_pid(ws, monkeypatch):
    stranger = subprocess.Popen(NAP, start_new_session=True)
    killed = []
    monkeypatch.setattr(jobs, "kill_group", killed.append)
    row = {"id": "old", "args": ["report"], "status": "running", "created": 0.0, "started": 0.0}
    JobStore(ws).save({**row, "ended": None, "code": None, "pid": stranger.pid})
    try:
        runner = Runner(ws, Hub(), parallel=1)
        runner.cancel("old")
        assert runner.jobs["old"].status == "interrupted" and stranger.poll() is None and killed == []
    finally:
        stranger.kill()
        stranger.wait()


def test_a_job_cancelled_while_waiting_for_admission_never_starts(ws, monkeypatch):
    gate, asked, spawned = [False], [], []
    monkeypatch.setattr(jobs, "ADMIT_POLL_S", POLL_SECONDS)
    monkeypatch.setattr(jobs, "can_start_job", lambda running: asked.append(running) or gate[0])
    monkeypatch.setattr(jobs, "start", lambda *args: spawned.append(args))
    runner = Runner(ws, Hub(), parallel=1)
    job = runner.submit(["report"])
    wait_for(lambda: asked)
    runner.cancel(job.id)
    gate[0] = True
    time.sleep(ADMIT_WAIT_S)
    assert job.status == "cancelled" and spawned == []


def test_cancel_kills_a_job_whose_pid_is_set_before_it_shows_running(ws, monkeypatch):
    killed = []
    monkeypatch.setattr(jobs, "kill_group", killed.append)
    runner = Runner(ws, Hub(), parallel=1)
    proc = subprocess.Popen(NAP, start_new_session=True)
    try:
        job = Job(id="gap", args=["report"], pid=proc.pid, started=time.time())
        runner.jobs[job.id], runner.procs[job.id] = job, proc
        runner.cancel(job.id)
        assert killed == [proc.pid] and job.status == "queued" and job.cancel
    finally:
        proc.kill()
        proc.wait()


def test_a_sleeping_machine_does_not_count_as_a_stalled_job(ws, monkeypatch):
    import byosm.app.jobs as jobs_module

    clock = [1000.0]
    monkeypatch.setattr(jobs_module.time, "time", lambda: clock[0])

    def sleep(seconds):
        clock[0] += 8 * 3600 if clock[0] < 1001 else seconds

    monkeypatch.setattr(jobs_module.time, "sleep", sleep)
    runner = Runner.__new__(Runner)
    runner.ws, runner.stopped, killed = ws, False, []
    job = Job(id="x", args=["build"], pid=123)
    calls = iter([False, False, True])
    monkeypatch.setattr(runner, "_tail", lambda job: None, raising=False)
    monkeypatch.setattr(runner, "_settled", lambda job, tail, heard: next(calls), raising=False)
    monkeypatch.setattr(runner, "_kill", killed.append, raising=False)
    monkeypatch.setattr(runner, "_line", lambda job, line: None, raising=False)
    runner._follow(job)
    assert killed == [] and not job.stalled


def test_jobs_keep_the_mac_awake_unless_switched_off(ws):
    import yaml

    from byosm.resources import keeps_awake

    assert keeps_awake(ws)
    cfg = yaml.safe_load((ws / "byosm.yaml").read_text())
    (ws / "byosm.yaml").write_text(yaml.safe_dump({**cfg, "keep_awake": False}))
    assert not keeps_awake(ws)


def test_a_silent_job_that_keeps_the_cpu_busy_is_not_stopped():
    import byosm.app.watchdog as watchdog

    cpu = iter([0.0, 120.0, 120.0])
    watch = watchdog.Watch(cpu=lambda: next(cpu), limit=10)
    watch.heard[0] = watch.last = __import__("time").time() - 20
    assert not watch.stuck()
    watch.heard[0] -= 20
    assert watch.stuck()
