import json

import numpy as np
import pytest

from byosm.games.poker import student
from byosm.games.poker.infosets import PAD, layout

WIDTH, HIST, BET = 3, 4, 4


def situations(n=600, seed=0):
    rng = np.random.default_rng(seed)
    rows = np.zeros(n, layout(WIDTH, HIST))
    rows["bucket"] = rng.integers(0, 12, n)
    rows["tokens"] = rng.integers(0, HIST + 1, n)
    rows["hist"] = rng.choice([0, 1, BET], (n, HIST))
    rows["hist"][np.arange(HIST)[None, :] >= rows["tokens"][:, None]] = PAD
    facing = rng.random(n) < 0.5
    rows["actions"] = np.where(facing, 2, 3)
    rows["offered"] = np.where(facing[:, None], [0, 1, PAD], [1, BET, 2])
    strong = rows["bucket"] >= 6
    rows["probs"] = np.where(strong[:, None], [0.1, 0.1, 0.8], [0.8, 0.1, 0.1])
    rows["probs"][facing] = np.where(strong[facing, None], [0.1, 0.9, 0.0], [0.9, 0.1, 0.0])
    rows["visits"] = rng.integers(0, 5, n)
    return rows


def test_a_student_learns_a_strategy_from_its_card_bucket_and_never_plays_a_move_not_on_offer(tmp_path, monkeypatch):
    rows = situations()
    said = []
    net, score = student.fit(rows, 12, said.append, epochs=120)
    whole = student.scored(net, rows[rows["visits"] > 0], "cpu")["kl"]
    assert score["top_move"] > 0.95 and score["kl"] < 0.05 and score["trained_on"] + score["held_out"] < len(rows)
    assert said and "held-out KL" in said[-1]
    student.table(net, rows, tmp_path / "probs", "cpu")
    got = np.fromfile(tmp_path / "probs", "<f4").reshape(len(rows), WIDTH)
    monkeypatch.setattr(student, "CHUNK", 64)
    student.table(net, rows, tmp_path / "in-pieces", "cpu")
    assert np.allclose(got, np.fromfile(tmp_path / "in-pieces", "<f4").reshape(len(rows), WIDTH), atol=1e-6)
    assert student.scored(net, rows[rows["visits"] > 0], "cpu")["kl"] == pytest.approx(whole, rel=1e-4)
    assert np.allclose(got.sum(1), 1, atol=1e-5) and (got[rows["offered"] == PAD] == 0).all()
    unseen = rows["visits"] == 0
    assert unseen.any() and (got[unseen].argmax(1) == rows["probs"][unseen].argmax(1)).mean() > 0.95


def test_the_same_seed_trains_the_same_student():
    rows = situations(200)
    first, _ = student.fit(rows, 12, lambda _: None, epochs=3)
    again, _ = student.fit(rows, 12, lambda _: None, epochs=3)
    assert all((a == b).all() for a, b in zip(first.state_dict().values(), again.state_dict().values(), strict=True))


def test_building_a_student_trains_it_writes_its_table_and_asks_the_engine_for_a_strategy_file(ws, monkeypatch):
    import contextlib

    from byosm.db.workspace import workspace_report
    from byosm.games.poker import distill, teacher

    rows = situations(300)
    folder = teacher.work_dir(ws)
    rows.tofile(folder / "old.infosets")
    (folder / "old.infosets.json").write_text(json.dumps({"records": len(rows), "width": WIDTH, "hist": HIST}))
    (folder / "old.bin").write_text("")
    asked, tables = [], []

    def engine(ws, hands, args, log):
        asked.append(args)
        if args[0] == "student":
            tables.append((folder / "old-student.probs").stat().st_size)

    monkeypatch.setattr(distill, "run", engine)
    monkeypatch.setattr("byosm.train.gpu.gpu_slot", lambda device, log: contextlib.nullcontext("cpu"))
    monkeypatch.setattr(student, "EPOCHS", 2)
    made = distill.build(ws, ws, "old", lambda _: None)
    assert made["student"] == "old-student" and made["teacher"] == "old" and made["model_mb"] > 0
    assert tables == [len(rows) * WIDTH * 4] and not (folder / "old-student.probs").exists()
    assert asked[0][:7] == ["student", "--like", "old.bin", "--records", "old.infosets", "--probs", "old-student.probs"]
    assert asked[0][7:] == ["--width", "3", "--hist", "4", "--out", "old-student.bin"]
    assert workspace_report(ws, "poker-student:old-student") == made


def test_extra_rounds_add_what_the_seated_student_reaches_and_refuse_an_unordered_export(ws, monkeypatch):
    import contextlib

    from byosm.games.poker import distill, infosets, teacher

    rows = situations(300)
    rows["visits"][:150] = 0
    folder = teacher.work_dir(ws)
    rows.tofile(folder / "old.infosets")
    meta = {"records": len(rows), "width": WIDTH, "hist": HIST}
    (folder / "old.infosets.json").write_text(json.dumps(meta))
    for name in ("old", "old-student-r1"):
        (folder / f"{name}.bin").write_text("")
    asked, trained_on = [], []

    def engine(ws, hands, args, log):
        asked.append(args[0])
        if args[0] == "reach":
            np.full(len(rows), 3, "<u4").tofile(folder / args[args.index("--out") + 1])

    fit = student.fit
    monkeypatch.setattr(distill, "run", engine)
    monkeypatch.setattr(infosets, "run", engine)
    monkeypatch.setattr("byosm.train.gpu.gpu_slot", lambda device, log: contextlib.nullcontext("cpu"))
    monkeypatch.setattr(student, "EPOCHS", 1)
    monkeypatch.setattr(student, "fit", lambda *a, **k: trained_on.append(int((k["visits"] > 0).sum())) or fit(*a, **k))
    with pytest.raises(ValueError, match="predates ordered exports"):
        distill.build(ws, ws, "old", lambda _: None, rounds=1)
    (folder / "old.infosets.json").write_text(json.dumps({**meta, "ordered": True}))
    asked.clear()
    trained_on.clear()
    made = distill.build(ws, ws, "old", lambda _: None, rounds=1)
    assert made["student"] == "old-student-r1" and made["rounds"] == 1
    assert asked == ["student", "reach", "student", "abstraction"] and trained_on[1] == len(rows) > trained_on[0]


class Engine:
    def __init__(self, answers):
        self.answers, self.asked = list(answers), []
        self.stdin = self.stdout = self

    def write(self, line):
        self.asked.append(json.loads(line))

    def flush(self):
        pass

    def readline(self):
        return json.dumps(self.answers.pop(0))


def test_advice_is_the_students_probabilities_over_the_moves_the_engine_offers():
    import torch

    from byosm.games.poker.advisor import Advisor

    torch.manual_seed(0)
    spot = {"seat": 2, "player": "BB", "street": "Flop", "bucket": 7, "hist": [0, 5, 15, 1, 5], "offered": [0, 1, 7, 2]}
    moves = ["fold", "call 300", "raise to 1450", "all in for 9750"]
    engine = Engine([{**spot, "moves": moves}, {"error": "the hand is already over"}])
    advisor = Advisor(student.Net(12).eval(), engine)
    found = advisor.advise("a hand")
    assert engine.asked == ["a hand"] and found["seat"] == 2 and found["player"] == "BB" and found["street"] == "flop"
    assert [m["move"] for m in found["moves"]] == moves
    assert sum(m["probability"] for m in found["moves"]) == pytest.approx(1, abs=1e-5)
    with pytest.raises(ValueError, match="already over"):
        advisor.advise("another")


def test_a_first_decision_with_no_history_is_advised_too():
    from byosm.games.poker.advisor import Advisor

    first = {"seat": 3, "player": "UTG", "street": "Preflop", "bucket": 6, "hist": []}
    spot = {**first, "offered": [0, 1, 4], "moves": ["a", "b", "c"]}
    found = Advisor(student.Net(169).eval(), Engine([spot])).advise("first in")
    assert sum(m["probability"] for m in found["moves"]) == pytest.approx(1, abs=1e-5)
