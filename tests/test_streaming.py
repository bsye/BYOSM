import random

from byosm.rt import TextGate
from byosm.system.streaming import replay, stream_summary, stream_units, tune
from byosm.text import prefix

GOLD = [{"input": "wake me up", "action": "alarm_set"}, {"input": "play jazz", "action": "play_music"}]
TRACES = [
    [[1, "weather_query", 0.3], [4, "alarm_set", 0.9], [5, "alarm_set", 0.95], [10, "alarm_set", 0.99]],
    [[1, "alarm_set", 0.97], [4, "play_music", 0.6], [9, "play_music", 0.99]],
]


def test_the_gate_fires_once_after_enough_sure_agreeing_steps():
    g = TextGate(min_confidence=0.8, stable_for=2)
    pushes = [("a", 0.9), ("b", 0.9), ("b", 0.5), ("b", 0.9), ("b", 0.9), ("c", 0.99), ("c", 0.99)]
    assert [g.push(label, c) for label, c in pushes] == [None, None, None, None, "b", None, None]


def test_replay_reports_the_typed_share_and_whether_it_fired_early():
    assert replay(TRACES[0], 10, {"min_confidence": 0.8, "stable_for": 2}) == ("alarm_set", 0.5, True)
    assert replay(TRACES[0], 10, None) == ("alarm_set", 1.0, False)


def test_tuning_keeps_wrong_fires_within_the_allowance():
    loose = stream_summary(stream_units(TRACES, GOLD, {"min_confidence": 0.5, "stable_for": 1}))
    assert loose["wrong_per_100"] == 50
    gate = tune(TRACES, GOLD, max_extra_wrong=1.0)
    s = stream_summary(stream_units(TRACES, GOLD, gate))
    assert s["wrong_per_100"] == 0 and s["typed_share"] < 1


def test_a_prefix_is_a_non_empty_start_of_the_text():
    rng = random.Random(0)
    cuts = {prefix("set an alarm", rng) for _ in range(200)}
    assert all("set an alarm".startswith(c) and c for c in cuts) and len(cuts) > 5
