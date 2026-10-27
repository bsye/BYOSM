import pytest

from byosm.pipeline import fixes_speed
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
