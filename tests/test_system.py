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


def test_a_system_board_becomes_a_report_block(ws):
    from byosm.report.systems import system_block

    split = {"exact_frame": 0.5, "action_accuracy": 0.9, "arg_f1": 0.7, "n": 2}
    mine = {"calib": split, "test": split, "ci": {"lo": 0.4, "hi": 0.6}, "latency_ms": 3.0}
    paired = {"llm": {s: {"delta": 0.2, "p_better": 1.0} for s in ("calib", "test")}}
    board = {"m": {"rows": {"calib": 1, "test": 2}, "rivals": {"system": mine, "llm": mine}, "paired": paired}}
    (ws / "systems" / "a").mkdir(parents=True)
    (ws / "systems" / "a" / "board.json").write_text(json.dumps(board))
    block = system_block("a")
    assert "| a | 0.500 [0.400, 0.600] |" in block and "a minus llm: test +0.200" in block


def test_a_router_only_team_scores_on_its_router_held_out_rows(ws):
    from byosm.data.store import Dataset, record
    from byosm.project import Project
    from byosm.spec import HFSource, Label, TaskSpec
    from byosm.system.frames import BUILTIN_SPLIT, _fetch
    from byosm.system.spec import Node, SystemSpec, SystemSuite

    spec = TaskSpec(name="r", description="d", modality="text", labels=[Label(name="a", definition="a")])
    Project("r").save_spec(spec)
    Dataset(Project("r").data_dir_for(spec)).save("test", [record("hi", "a", "t"), record("yo", "a", "t")])
    node = Node(specialist="r", role="router", task="t", source=HFSource(dataset="x"))
    team = SystemSpec(name="t", description="d", nodes={"r": node})
    rows = _fetch(team, SystemSuite(name="held", split=BUILTIN_SPLIT, max_items=1))
    assert rows == [{"input": "hi", "action": "a", "args": []}]
