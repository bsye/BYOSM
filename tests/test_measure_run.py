import time

from conftest import promoted_specialist

from byosm.measure import conduct
from byosm.measure.conduct import MAX_JOBS, MEASURE_EVENT, Conductor, tick
from byosm.measure.run import DONE, FAILED, OPEN, PENDING, RUNNING, TODO, board, queue, save_run, saved_run

TIMEOUT_S = 10
POLL_S = 0.01


class FakeJob:
    def __init__(self, job_id, args, status="queued"):
        self.id, self.args, self.status = job_id, list(args), status


class FakeRunner:
    def __init__(self, *running):
        self.jobs = {j.id: j for j in running}
        self.started, self.cancelled = [], []

    def submit(self, args):
        job = FakeJob(f"j{len(self.jobs)}", args)
        self.jobs[job.id] = job
        self.started.append(job.args)
        return job

    def snapshot(self):
        return list(self.jobs.values())

    def cancel(self, job_id):
        self.cancelled.append(job_id)


def _cell(ws, specialist, measurement):
    return next(c for c in saved_run(ws)["cells"] if (c["specialist"], c["measurement"]) == (specialist, measurement))


def _job_ends(ws, runner, specialist, measurement, status):
    runner.jobs[_cell(ws, specialist, measurement)["job"]].status = status


def _seeds_measured(monkeypatch, name):
    from byosm import health
    from byosm.health_score import OK, check

    real = health.seeds_check

    def seeds(p, prim):
        return check("seeds", "Seed noise measured", 1, OK, "sd 0.01", "measured") if p.name == name else real(p, prim)

    monkeypatch.setattr(health, "seeds_check", seeds)


def _statuses(ws, runner):
    return {(c["specialist"], c["measurement"]): c["status"] for c in board(ws, runner.jobs)["cells"]}


def test_cpu_checks_start_at_once_and_gpu_trainings_wait_for_the_previous_specialist(ws, monkeypatch):
    promoted_specialist("qa")
    promoted_specialist("vad")
    runner = FakeRunner()
    queue(ws)
    assert tick(ws, runner)
    assert runner.started == [["overfit", "qa"], ["overfit", "vad"], ["seeds", "qa"]]
    _job_ends(ws, runner, "qa", "seeds", "running")
    assert not tick(ws, runner)
    _job_ends(ws, runner, "qa", "seeds", "done")
    _seeds_measured(monkeypatch, "qa")
    assert tick(ws, runner)
    assert runner.started[-1] == ["seeds", "vad"] and _cell(ws, "qa", "seeds")["state"] == DONE
    assert not runner.cancelled


def test_a_job_already_running_the_same_command_is_followed_not_started_again(ws):
    promoted_specialist()
    runner = FakeRunner(FakeJob("mine", ["overfit", "qa"], "running"))
    queue(ws)
    tick(ws, runner)
    assert runner.started == [["seeds", "qa"]]
    assert _cell(ws, "qa", "overfit")["job"] == "mine" and _statuses(ws, runner)[("qa", "overfit")] == RUNNING


def test_a_failed_job_fails_its_cell_and_hands_the_gpu_to_the_next_specialist(ws):
    promoted_specialist("qa")
    promoted_specialist("vad")
    runner = FakeRunner()
    queue(ws)
    tick(ws, runner)
    _job_ends(ws, runner, "qa", "seeds", "interrupted")
    tick(ws, runner)
    assert _statuses(ws, runner)[("qa", "seeds")] == FAILED and _cell(ws, "qa", "seeds")["note"] == "job interrupted"
    assert runner.started[-1] == ["seeds", "vad"] and not runner.cancelled


def test_a_measurement_still_missing_after_its_jobs_fails_instead_of_running_forever(ws):
    promoted_specialist()
    runner = FakeRunner()
    queue(ws)
    for _ in range(MAX_JOBS):
        tick(ws, runner)
        _job_ends(ws, runner, "qa", "seeds", "done")
    tick(ws, runner)
    cell = _cell(ws, "qa", "seeds")
    assert cell["state"] == FAILED and cell["note"] == f"still missing after {MAX_JOBS} jobs"
    assert runner.started.count(["seeds", "qa"]) == MAX_JOBS


def test_the_board_shows_what_is_missing_then_each_cell_as_its_job_moves(ws):
    p = promoted_specialist()
    runner = FakeRunner()
    before = board(ws, runner.jobs)
    assert before["progress"] == {"done": 0, "failed": 0, "todo": 2, "total": 2}
    assert [m["id"] for m in before["measurements"]][:2] == ["overfit", "public-suites"]
    assert set(_statuses(ws, runner).values()) == {TODO}
    queue(ws)
    tick(ws, runner)
    assert set(_statuses(ws, runner).values()) == {PENDING}
    _job_ends(ws, runner, "qa", "overfit", "running")
    assert _statuses(ws, runner) == {("qa", "overfit"): RUNNING, ("qa", "seeds"): PENDING}
    _job_ends(ws, runner, "qa", "overfit", "done")
    p.save_report("overfit", {"experiment": "v1", "verdict": "ok", "summary": "train and test agree"})
    tick(ws, runner)
    assert _statuses(ws, runner)[("qa", "overfit")] == DONE
    assert board(ws, runner.jobs)["progress"] == {"done": 1, "failed": 0, "todo": 0, "total": 2}


def test_queueing_again_keeps_open_cells_and_retries_failed_ones(ws):
    promoted_specialist()
    runner = FakeRunner()
    queue(ws)
    tick(ws, runner)
    queue(ws)
    tick(ws, runner)
    assert len(saved_run(ws)["cells"]) == 2 and runner.started == [["overfit", "qa"], ["seeds", "qa"]]
    _job_ends(ws, runner, "qa", "overfit", "failed")
    tick(ws, runner)
    assert _statuses(ws, runner)[("qa", "overfit")] == FAILED
    queue(ws)
    tick(ws, runner)
    assert _cell(ws, "qa", "overfit")["state"] == OPEN and runner.started.count(["overfit", "qa"]) == 2


def test_a_run_changed_meanwhile_is_not_overwritten(ws):
    promoted_specialist()
    queue(ws)
    stale = saved_run(ws)
    assert save_run(ws, saved_run(ws))
    assert not save_run(ws, stale)


def _wait_for(check):
    end = time.time() + TIMEOUT_S
    while not check():
        assert time.time() < end, "timed out"
        time.sleep(POLL_S)


def test_the_conductor_starts_queued_measurements_in_the_background_and_announces_them(ws, monkeypatch):
    monkeypatch.setattr(conduct, "TICK_S", POLL_S)
    promoted_specialist()
    runner, events = FakeRunner(), []
    conductor = Conductor(ws, runner, events.append)
    conductor.queue(None)
    _wait_for(lambda: len(runner.started) == 2 and MEASURE_EVENT in events)
    monkeypatch.setattr(conduct, "tick", lambda ws, runner: 1 / 0)
    _wait_for(lambda: conductor.error is not None)
    assert conductor.error.startswith("ZeroDivisionError")
    conductor.stop()
