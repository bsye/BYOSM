from __future__ import annotations

import json
import sys

import numpy as np
import pytest

from byosm.games.poker import seat, teacher
from byosm.games.poker.infosets import PAD, layout
from byosm.games.poker.portable import exported

REQUEST = {
    "id": 7,
    "seat": 1,
    "players": 6,
    "hole": ["Ks", "Qh"],
    "board": ["4d", "6s", "Th"],
    "street": 1,
    "sb": 50,
    "bb": 100,
    "stacks": [9950, 9750, 10000, 10000, 9750, 10000],
    "contrib": [50, 250, 0, 0, 250, 0],
    "history": [
        {"seat": 2, "street": 0, "kind": "fold", "to": 0},
        {"seat": 3, "street": 0, "kind": "fold", "to": 0},
        {"seat": 4, "street": 0, "kind": "raise", "to": 250},
        {"seat": 5, "street": 0, "kind": "fold", "to": 0},
        {"seat": 0, "street": 0, "kind": "fold", "to": 0},
        {"seat": 1, "street": 0, "kind": "call", "to": 0},
    ],
}
SPOT = {
    "bucket": 3,
    "hist": [0, 0, 5, 0, 0, 1, 15],
    "offered": [1, 4, 6, 2],
    "moves": ["check", "raise to 150", "raise to 400", "all in for 9750"],
}
ENGINE = "import sys\nfor line in sys.stdin:\n    print(sys.argv[1], flush=True)\n"


@pytest.fixture
def tiny(ws):
    import torch

    from byosm.games.poker.student import Net

    torch.manual_seed(0)
    net = Net(5)
    torch.save(net.state_dict(), teacher.work_dir(ws) / "tiny-student.pt")
    graph, meta = exported(ws, "tiny-student")
    return net, graph, json.loads(meta.read_text())


def test_a_table_request_becomes_the_hand_history_the_engine_reads():
    text = seat.hand(REQUEST)
    assert "starting_stacks = [10000, 10000, 10000, 10000, 10000, 10000]" in text
    assert "blinds_or_straddles = [50, 100, 0, 0, 0, 0]" in text and "min_bet = 100" in text
    assert seat.actions(REQUEST) == [
        "d dh p1 ????",
        "d dh p2 KsQh",
        "d dh p3 ????",
        "d dh p4 ????",
        "d dh p5 ????",
        "d dh p6 ????",
        "p3 f",
        "p4 f",
        "p5 cbr 250",
        "p6 f",
        "p1 f",
        "p2 cc",
        "d db 4d6sTh",
    ]
    later = {**REQUEST, "board": ["4d", "6s", "Th", "2c", "9h"], "street": 3}
    later["history"] = [*REQUEST["history"], {"seat": 1, "street": 1, "kind": "check", "to": 0}]
    assert seat.actions(later)[-4:] == ["d db 4d6sTh", "p2 cc", "d db 2c", "d db 9h"]


def test_each_move_the_engine_names_becomes_a_table_action():
    said = {"fold": "fold", "check": "check", "call 150": "call", "all in for 9750": "allin"}
    assert {move: seat.reply(REQUEST, move)["action"] for move in said} == said
    assert seat.reply(REQUEST, "raise to 400") == {"id": 7, "action": "raise", "to": 400}


def test_the_seat_plays_the_same_probabilities_as_the_student(tiny):
    import onnxruntime
    import torch

    from byosm.games.poker.student import inputs

    net, graph, meta = tiny
    session = onnxruntime.InferenceSession(str(graph), providers=["CPUExecutionProvider"])
    row = np.zeros(1, layout(len(SPOT["offered"]), len(SPOT["hist"])))
    row["bucket"], row["actions"], row["tokens"] = SPOT["bucket"], len(SPOT["offered"]), len(SPOT["hist"])
    row["offered"][0], row["hist"][0] = SPOT["offered"], PAD
    row["hist"][0, : len(SPOT["hist"])] = SPOT["hist"]
    with torch.no_grad():
        expected = torch.softmax(net.eval()(**inputs(row, net.span, "cpu")), dim=-1)[0].numpy()[SPOT["offered"]]
    played = seat.chances(session.run(None, seat.row(SPOT, meta))[0], SPOT["offered"])
    assert np.allclose(played, expected, atol=1e-5) and abs(played.sum() - 1) < 1e-9


def test_a_seat_answers_a_request_the_same_way_twice_and_passes_on_a_refusal(tiny):
    _, graph, meta = tiny
    playing = seat.Seat([sys.executable, "-c", ENGINE, json.dumps(SPOT)], str(graph), meta, seed=1)
    first = playing.act(REQUEST)
    assert first == playing.act(REQUEST) and first["id"] == 7
    assert first["action"] in ("check", "raise", "allin")
    refusing = seat.Seat([sys.executable, "-c", ENGINE, json.dumps({"error": "out of turn"})], str(graph), meta, 1)
    assert refusing.act(REQUEST) == {"id": 7, "error": "out of turn"}
