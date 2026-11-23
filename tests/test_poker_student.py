import numpy as np

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


def test_a_student_learns_a_strategy_from_its_card_bucket_and_never_plays_a_move_not_on_offer(tmp_path):
    rows = situations()
    said = []
    net, score = student.fit(rows, 12, said.append, epochs=120)
    assert score["top_move"] > 0.95 and score["kl"] < 0.05 and score["trained_on"] + score["held_out"] < len(rows)
    assert said and "held-out KL" in said[-1]
    student.table(net, rows, tmp_path / "probs", "cpu")
    got = np.fromfile(tmp_path / "probs", "<f4").reshape(len(rows), WIDTH)
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
    import json

    from byosm.db.workspace import workspace_report
    from byosm.games.poker import distill, teacher

    rows = situations(300)
    folder = teacher.work_dir(ws)
    rows.tofile(folder / "old.infosets")
    (folder / "old.infosets.json").write_text(json.dumps({"records": len(rows), "width": WIDTH, "hist": HIST}))
    (folder / "old.bin").write_text("")
    asked = []
    monkeypatch.setattr(distill, "run", lambda ws, hands, args, log: asked.append(args))
    monkeypatch.setattr("byosm.train.gpu.gpu_slot", lambda device, log: contextlib.nullcontext("cpu"))
    made = distill.build(ws, ws, "old", lambda _: None)
    assert made["student"] == "old-student" and made["teacher"] == "old" and made["model_mb"] > 0
    assert (folder / "old.student.probs").stat().st_size == len(rows) * WIDTH * 4
    assert asked[0][:7] == ["student", "--like", "old.bin", "--records", "old.infosets", "--probs", "old.student.probs"]
    assert asked[0][7:] == ["--width", "3", "--hist", "4", "--out", "old-student.bin"]
    assert workspace_report(ws, "poker-student:old-student") == made
