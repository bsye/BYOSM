from pathlib import Path

import pytest

from byosm.promotion import fixes_speed
from byosm.spec import Label, TaskSpec

TOLERANCE = 0.01
SLOW, FAST = {"latency_ms": 163.0}, {"latency_ms": 10.0}


@pytest.mark.parametrize(
    ("mine", "cur", "value", "promoted"),
    [
        (FAST, SLOW, 0.978 - TOLERANCE, True),
        (FAST, SLOW, 0.956, False),
        (FAST, FAST, 0.978, False),
        ({"latency_ms": 150.0}, SLOW, 0.978, False),
        ({}, SLOW, 0.978, False),
        (FAST, {**SLOW, "latency_load": 3.0}, 0.978 - TOLERANCE, False),
        ({**FAST, "latency_load": 0.4}, {**SLOW, "latency_load": 0.4}, 0.978 - TOLERANCE, True),
    ],
)
def test_a_model_is_promoted_for_speed_only_when_it_meets_the_budget_the_current_one_misses(mine, cur, value, promoted):
    labels = [Label(name="a", definition=""), Label(name="b", definition="")]
    spec = TaskSpec(name="gate", description="", modality="text", labels=labels, latency_budget_ms=100)
    assert fixes_speed(spec, mine, cur, value, 0.978) is promoted


def test_a_specialist_can_trade_more_accuracy_for_meeting_its_latency_budget():
    labels = [Label(name="a", definition=""), Label(name="b", definition="")]
    spec = TaskSpec(name="tc", description="", modality="text", labels=labels, latency_budget_ms=100)
    assert not fixes_speed(spec, FAST, SLOW, 0.956, 0.978)
    spec.speed_tolerance = 0.025
    assert fixes_speed(spec, FAST, SLOW, 0.956, 0.978)


def test_a_faster_student_is_promoted_from_the_board_when_it_is_within_the_tolerance(ws):
    from conftest import save_board

    from byosm.project import Project
    from byosm.promotion import maybe_promote

    p = Project("tc").ensure()
    labels = [Label(name="a", definition=""), Label(name="b", definition="")]
    spec = TaskSpec(
        name="tc", description="", modality="text", labels=labels, latency_budget_ms=100, speed_tolerance=0.025
    )
    p.save_spec(spec)
    for exp in ("v1", "distill-v1"):
        p.save_experiment(exp, meta={"calib_held": True})
    p.promote("v1")

    def result(calib, ms):
        return {
            "kind": "experiment",
            "objective": calib,
            "calib_objective": calib,
            "n_test": 500,
            "n_calib": 200,
            "latency_ms": ms,
        }

    save_board(p, {"v1": {"builtin": result(0.978, 163.0)}, "distill-v1": {"builtin": result(0.956, 10.3)}})
    maybe_promote(p, "distill-v1", lambda _: None)
    assert p.promoted == "distill-v1"


def test_a_latency_timed_while_the_machine_was_busy_is_unknown_not_slow(monkeypatch):
    from byosm.evals import scoring
    from byosm.health import _speed

    labels = [Label(name="a", definition=""), Label(name="b", definition="")]
    spec = TaskSpec(name="gate", description="", modality="text", labels=labels, latency_budget_ms=100)
    assert _speed(spec, {**SLOW, "latency_load": 0.3})["status"] == "warn"
    busy = _speed(spec, {**SLOW, "latency_load": 2.5})
    assert busy["status"] == "unknown" and "busy" in busy["value"]

    class Timed:
        def stream_chunk_ms(self):
            return None

        def latency_ms(self, x):
            return 12.0

    monkeypatch.setattr("byosm.machine.load_per_core", lambda: 1.7)
    monkeypatch.setattr(scoring, "get_modality", lambda spec, root: type("M", (), {"load": staticmethod(lambda r: r)}))
    out = {}
    scoring._add_speed(out, Timed(), spec, [{"input": "x"}], Path("."))
    assert out == {"latency_ms": 12.0, "latency_load": 1.7} and scoring.fair_ms(out) is None
