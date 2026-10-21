import json

from byosm.rt import System
from byosm.system.frames import exact_bootstrap, summary, units

GOLD = [
    {"action": "alarm_set", "args": [["time", "five am"]]},
    {"action": "play_music", "args": []},
    {"action": "weather_query", "args": [["place_name", "Paris"]]},
]


def test_a_frame_is_exact_only_when_the_action_and_every_argument_match():
    pred = [
        {"action": "alarm_set", "args": [["time", "Five AM"]]},
        {"action": "play_music", "args": [["song_name", "x"]]},
        {"action": "calendar_set", "args": [["place_name", "paris"]]},
    ]
    u = units(pred, GOLD)
    assert u["exact"] == [1, 0, 0] and u["action"] == [1, 1, 0]
    s = summary(u)
    assert s["arg_f1"] == 2 * 2 / (2 * 2 + 1 + 0) and abs(s["exact_frame"] - 1 / 3) < 1e-9
    assert exact_bootstrap(u, units(GOLD, GOLD))["delta"] < 0


def test_tool_arguments_are_filled_from_tagged_types(tmp_path):
    manifest = {"nodes": {}, "tools": {"set_alarm": {"when": "alarm_set", "args": {"at": "time", "day": "date"}}}}
    (tmp_path / "system.json").write_text(json.dumps(manifest))
    system = System(tmp_path)
    assert system._call_args("set_alarm", [{"type": "time", "text": "five am"}]) == {"at": "five am", "day": None}
