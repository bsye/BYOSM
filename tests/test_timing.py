import pytest
from conftest import save_board

from byosm.evals import timing
from byosm.health import _speed
from byosm.project import Project
from byosm.spec import Label, TaskSpec

LABELS = [Label(name=n, definition="") for n in "ab"]
QUIET, BUSY = 0.2, 2.5


@pytest.fixture
def timed(ws, monkeypatch):
    p = Project("t").ensure()
    p.save_spec(TaskSpec(name="t", description="", modality="text", labels=LABELS, latency_budget_ms=100))
    for e in ("slow", "fair"):
        p.save_experiment(e, meta={"params": 1})
    busy = {"kind": "experiment", "objective": 0.9, "latency_ms": 650.0, "latency_load": BUSY}
    save_board(p, {"slow": {"builtin": busy}, "fair": {"builtin": {**busy, "latency_ms": 90.0, "latency_load": QUIET}}})
    monkeypatch.setattr(timing, "materialize", lambda p, spec, suite, log: ([], [{"input": "x"}], ws))
    monkeypatch.setattr(timing, "experiment_predictor", lambda p, spec, exp: object())
    monkeypatch.setattr(
        timing, "add_speed", lambda out, pred, spec, test, root: out.update(latency_ms=95.0, latency_load=QUIET)
    )
    return p


def test_only_a_timing_taken_while_busy_is_taken_again_and_its_other_scores_are_kept(timed, monkeypatch):
    monkeypatch.setattr(timing, "load_per_core", lambda: QUIET)
    assert timing.busy_timings(timed) == ["slow"]
    assert timing.retime("t", log=lambda _: None) == {"slow": {"latency_ms": 95.0, "latency_load": QUIET}}
    again = timed.result("slow", "builtin")
    assert again["latency_ms"] == 95.0 and again["objective"] == 0.9 and timing.busy_timings(timed) == []


def test_a_machine_that_stays_busy_is_not_timed_at_all(timed, monkeypatch):
    monkeypatch.setattr(timing, "load_per_core", lambda: BUSY)
    with pytest.raises(RuntimeError):
        timing.wait_quiet(lambda _: None, wait_s=0)
    assert timed.result("slow", "builtin")["latency_ms"] == 650.0


def test_a_busy_timing_becomes_a_step_that_times_it_again(timed):
    check = _speed(timed.spec(), timed.result("slow", "builtin"))
    assert check["status"] == "unknown" and check["args"] == ["time", "t"]


def test_a_specialist_with_a_busy_timing_lists_time_it_again_among_its_next_steps(timed):
    from conftest import save_board

    from byosm.next_steps import suggestions

    timed.promote("slow", "test")
    save_board(timed, {"slow": {"builtin": {**timed.result("slow", "builtin"), "calib_objective": 0.9}}})
    step = next(s for s in suggestions("t") if s["id"] == "speed")
    assert step["title"] == "Time it again" and step["args"] == ["time", "t"]
