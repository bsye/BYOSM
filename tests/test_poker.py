import csv

import numpy as np
import pytest
from typer.testing import CliRunner

from byosm import metrics, routing
from byosm.benchmarks import pluribus, pokerbench
from byosm.benchmarks.catalog import fitting
from byosm.cli import app
from byosm.games.poker import create as creating
from byosm.games.poker import facts, lines
from byosm.games.poker.moves import LABELS, label_of, labelled
from byosm.games.poker.pluribus import decisions
from byosm.games.poker.spots import Raw, postflop_spot, preflop_spot
from byosm.project import Project
from byosm.spec import Label, TaskSpec

PREFLOP = {
    "prev_line": "UTG/2.0bb/BTN/call/SB/13.0bb/BB/allin/UTG/fold/BTN/fold",
    "hero_pos": "SB",
    "hero_holding": "KdKc",
    "correct_decision": "call",
    "available_moves": "['call', 'fold']",
    "pot_size": "117.0",
}
POSTFLOP = {
    "preflop_action": "HJ/2.0bb/BB/call",
    "board_flop": "Ks7h2d",
    "board_turn": "Jc",
    "board_river": "7c",
    "aggressor_position": "OOP",
    "postflop_action": "OOP_CHECK/IP_CHECK/dealcards/Jc/OOP_CHECK/IP_BET_5/OOP_RAISE_14/IP_CALL/dealcards/7c/OOP_CHECK",
    "evaluation_at": "River",
    "available_moves": "['Check', 'Bet 24', 'Bet 12']",
    "pot_size": "32",
    "hero_position": "IP",
    "holding": "8h8c",
    "correct_decision": "Bet 24",
}


def test_sized_moves_are_labelled_by_rank_among_the_offered_sizes_of_their_kind():
    assert labelled(["5.0bb", "3.0bb", "check", "fold"]) == {
        "5.0bb": "raise big",
        "3.0bb": "raise small",
        "check": "check",
        "fold": "fold",
    }
    assert label_of("Bet 24", ["Check", "Bet 24"]) == "bet small"
    assert label_of("allin", ["allin", "26.0bb", "call", "fold"]) == "allin"


def test_a_move_the_labels_cannot_hold_fails_loudly():
    with pytest.raises(ValueError, match="3 raise sizes"):
        labelled(["Raise 10", "Raise 20", "Raise 30"])
    with pytest.raises(ValueError, match="unknown poker move"):
        labelled(["limp"])
    with pytest.raises(ValueError, match="not among the offered"):
        label_of("call", ["check", "fold"])


def test_a_spot_reads_as_the_action_street_by_street_with_the_offered_moves():
    pre = preflop_spot(PREFLOP)
    assert pre.label == "call" and pre.street == "preflop"
    assert "UTG raise 2.0bb, BTN call, SB raise 13.0bb, BB all in" in pre.text
    assert pre.text.endswith("Moves: call, fold.")
    post = postflop_spot(POSTFLOP)
    assert post.label == "bet big" and post.street == "river"
    assert "Turn Jc: OOP check, IP bet 5, OOP raise 14, IP call. River 7c: OOP check." in post.text
    assert "holding 8h 8c" in post.text


def spec(families=True):
    labels = [Label(name=n, definition=n, family=n.split()[0] if families else None) for n in LABELS]
    return TaskSpec(name="p", description="", modality="text", labels=labels, positive_label="fold")


def test_family_accuracy_counts_a_move_of_the_right_kind_at_the_wrong_size_as_right():
    s = spec()
    y = np.array([LABELS.index("bet small"), LABELS.index("call")])
    probs = np.eye(len(LABELS))[[LABELS.index("bet big"), LABELS.index("fold")]]
    m = metrics.compute(s, y, probs, None)
    assert m["accuracy"] == 0.0 and m["family_accuracy"] == 0.5
    assert "family_accuracy" not in metrics.compute(spec(families=False), y, probs, None)


def test_labels_either_all_name_a_family_or_none_do():
    s = spec()
    s.labels[0].family = None
    with pytest.raises(ValueError, match="every label names a family"):
        assert s.label_families


def test_a_family_field_left_empty_keeps_the_data_key():
    s = spec(families=False)
    s.data.teachers = ["llm"]
    before = s.data_key()
    s.labels[0].family = None
    assert s.data_key() == before and "family" not in s.labels_dump()[0]


def write(path, rows):
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return path


def hub(tmp_path, monkeypatch):
    other = {**POSTFLOP, "holding": "AhAd", "correct_decision": "Check"}
    files = {
        "preflop": write(
            tmp_path / "pre.csv", [PREFLOP, {**PREFLOP, "hero_holding": "2c7d", "correct_decision": "fold"}]
        ),
        "postflop": write(tmp_path / "post.csv", [POSTFLOP, other] * 20),
    }
    monkeypatch.setattr(pokerbench, "download", lambda repo, file: files[file.partition("_")[0]])
    monkeypatch.setattr("byosm.data.hf_source.cards.licence", lambda dataset: "apache-2.0")
    hands = []
    for i in range(10):
        hands.append(tmp_path / f"{i}.phh")
        hands[-1].write_text(HAND)
    monkeypatch.setattr(pluribus, "hands", lambda: hands)
    monkeypatch.setattr(pluribus, "DECISIONS", 30)


def test_pokerbench_fits_only_the_move_labels_and_its_floor_scores_the_label_share(tmp_path, monkeypatch):
    hub(tmp_path, monkeypatch)
    assert [b.name for b in fitting(spec())] == ["pokerbench", "pluribus"] and fitting(spec(families=False)) == []
    rows = pokerbench.rows(spec(), print)
    test = [r for r in rows if r["split"] == "test"]
    found = pokerbench.expected(spec(), rows)
    assert found["accuracy"] == sum(r["label"] == "fold" for r in test) / len(test)
    assert pokerbench.observed({"metrics": {"accuracy": 0.2, "family_accuracy": 0.3}}) == {
        "accuracy": 0.2,
        "family_accuracy": 0.3,
    }


def test_training_spots_keep_the_natural_move_mix_and_never_copy_a_test_spot():
    other = {**POSTFLOP, "holding": "AhAd", "correct_decision": "Check"}
    pre, post, again = Raw(PREFLOP, True, "call"), Raw(POSTFLOP, False, "bet big"), Raw(other, False, "check")
    copy = Raw({**PREFLOP, "correct_decision": "fold"}, True, "fold")
    picked = creating.sampled([pre, post, again, again], [copy], len(LABELS))
    assert pre not in picked and sorted(s.label for s in picked) == ["bet big", "check", "check"]


def test_poker_new_builds_the_specialist_with_pokerbench_deciding_promotion(ws, monkeypatch):
    hub(ws, monkeypatch)
    result = CliRunner().invoke(app, ["poker", "new", "holdem"])
    assert result.exit_code == 0, result.output
    s = Project("holdem").spec()
    assert s.primary_suite() == "pokerbench" and s.objective.metric == "accuracy"
    assert s.selecting() == ["pokerbench", "pluribus"]
    assert s.label_families == [n.split()[0] for n in LABELS]
    assert "protocol check, accuracy" in result.output
    again = CliRunner().invoke(app, ["poker", "new", "holdem"])
    assert again.exit_code != 0 and "already exists" in again.output
    asked = []
    table = creating.build_table
    monkeypatch.setattr(
        creating, "build_table", lambda ws, name, size, log: asked.append(size) or table(ws, name, size, log)
    )
    fewer = CliRunner().invoke(app, ["poker", "new", "small", "--spots", "1"])
    assert fewer.exit_code == 0 and asked == [1]


def test_a_poker_request_in_home_creates_the_poker_specialist_and_others_go_on_as_before(ws, monkeypatch):
    hub(ws, monkeypatch)
    monkeypatch.setattr(routing, "route", lambda prompt, cfg: {"kind": routing.POKER})
    assert routing.routed("what should I do with KK facing a shove", print) == (routing.POKER, "poker-move")
    assert Project("poker-move").spec().description == "what should I do with KK facing a shove"
    monkeypatch.setattr(routing, "route", lambda prompt, cfg: {"kind": routing.SPECIALIST})
    assert routing.routed("is this email spam", print) == (routing.SPECIALIST, None)


HAND = """variant = 'NT'
ante_trimming_status = true
antes = [0, 0, 0, 0, 0, 0]
blinds_or_straddles = [50, 100, 0, 0, 0, 0]
min_bet = 100
starting_stacks = [10000, 10000, 10000, 10000, 10000, 10000]
actions = ['d dh p1 TcQc', 'd dh p2 KsQh', 'd dh p3 9c3d', 'd dh p4 Ah4h', 'd dh p5 Th5s', 'd dh p6 6c7s', \
'p3 f', 'p4 f', 'p5 cbr 250', 'p6 f', 'p1 f', 'p2 cc', 'd db 4d6sTh', 'p2 cc', 'p5 cbr 300', 'p2 cbr 9750', 'p5 f']
hand = 3
players = ['MrBlue', 'Pluribus', 'MrWhite', 'MrPink', 'MrBrown', 'MrOrange']
finishing_stacks = [9950, 10550, 10000, 10000, 9500, 10000]
"""


def test_each_pluribus_decision_becomes_a_spot_rendered_like_pokerbench():
    found = list(decisions(HAND))
    assert [s.label for s in found] == ["call", "check", "allin"]
    assert found[0].text == (
        "Preflop, 6-max, 100 big blinds deep. You are BB holding Ks Qh. "
        "Action: UTG fold, HJ fold, CO raise 2.5bb, BTN fold, SB fold. Pot 4. Facts: KQ offsuit; 60% equity against 1 "
        "random hand; 1.5 to call, pot odds 27%. Moves: fold, call, raise, allin."
    )
    assert "You are OOP holding Ks Qh, IP was the last aggressor." in found[1].text
    assert found[1].text.endswith(
        "Flop 4d 6s Th: no action yet. Pot 5.5. Facts: high card; 45% equity against 1 random hand. Moves: check, bet, allin."
    )
    assert (
        "Flop 4d 6s Th: OOP check, IP bet 3. Pot 8.5." in found[2].text and "3 to call, pot odds 26%" in found[2].text
    )


def test_the_amount_to_call_and_the_players_still_in_follow_the_action():
    assert lines.preflop_state("", "UTG") == (1.0, 5)
    assert lines.preflop_state("UTG/fold/HJ/2.5bb", "BTN") == (2.5, 3)
    assert lines.preflop_state("HJ/2.0bb/CO/6.5bb/SB/allin", "HJ") == (98.0, 2)
    assert lines.postflop_state("HJ/2.0bb/BB/call", "OOP_CHECK/IP_BET_5/OOP_RAISE_14", "IP") == (9.0, 1)
    multiway = lines.postflop_state("UTG/2.0bb/HJ/call/CO/call/BTN/fold/SB/fold/BB/fold", "UTG_BET_3/HJ_FOLD", "CO")
    assert multiway == (3.0, 1)


def test_facts_name_the_hand_its_draws_its_equity_and_the_price_to_call():
    assert facts.preflop_class(["7s", "As"]) == "A7 suited" and facts.preflop_class(["Kd", "Kc"]) == "pocket KK"
    assert facts.draws(["Ah", "Kh"], ["Qh", "7h", "2c"]) == ["flush draw"]
    assert facts.draws(["9c", "8d"], ["7h", "6s", "2c"]) == ["open-ended straight draw"]
    assert facts.draws(["9c", "8d"], ["Jh", "7s", "2c"]) == ["gutshot"]
    assert facts.made(["2d", "3c"], ["Ah", "Ad", "Ks", "Kc", "Qh"]) == "only the board's two pair"
    said = facts.sentence(["Kd", "Kc"], [], 1, 87.0, 117.0)
    assert said == facts.sentence(["Kd", "Kc"], [], 1, 87.0, 117.0)
    assert said.startswith("Facts: pocket KK; 8") and said.endswith("87 to call, pot odds 43%.")
    assert "to call" not in facts.sentence(["Kd", "Kc"], ["2c", "7d", "9h"], 2, 0.0, 10.0)


def test_pluribus_hands_go_whole_to_training_or_to_the_suite_and_a_wrong_count_stops_the_replay(tmp_path, monkeypatch):
    hub(tmp_path, monkeypatch)
    monkeypatch.setattr(pluribus, "trains_on", lambda hand: hand.endswith(("/0", "/1", "/2")))
    held = pluribus.rows(spec(), print)
    assert len(pluribus.train_spots()) == 9 and len(held) == 21
    assert all(not r["meta"]["hand"].endswith(("/0", "/1", "/2")) for r in held)
    monkeypatch.setattr(pluribus, "DECISIONS", 31)
    with pytest.raises(ValueError, match="found 30 Pluribus decisions"):
        pluribus.rows(spec(), print)


def test_templated_inputs_drop_only_exact_copies_from_held_out_splits():
    from byosm.data import near_limit
    from byosm.data.store import dedupe, record

    s = spec()
    words = " ".join(f"w{i}" for i in range(40))
    train = [record(f"{words} Ks", "check", "t")]
    near = record(f"{words} Kd", "check", "t")
    assert dedupe([near], against=train, near=near_limit(s)) == []
    s.data.drop_near_copies = False
    assert dedupe([near, train[0]], against=train, near=near_limit(s)) == [near]


def test_the_teacher_runs_sealed_with_only_its_folder_and_the_hands_mounted(ws):
    from byosm.games.poker import teacher

    found = teacher.command(ws, ws / "hands", ["benchmark"])
    assert found[found.index("--network") : found.index("--network") + 2] == ["--network", "none"]
    assert "--read-only" in found and "--cap-drop" in found and f"{ws / 'hands'}:{teacher.HANDS}:ro" in found
    assert found[-2:] == [teacher.IMAGE, "benchmark"]


def test_teacher_training_resumes_from_its_checkpoint(ws, monkeypatch):
    from byosm.games.poker import teacher

    calls = []
    monkeypatch.setattr(teacher, "run", lambda ws, hands, args, log: calls.append(args))
    teacher.train(ws, ws, 1000, print)
    (teacher.work_dir(ws) / f"{teacher.blueprint(1000)}{teacher.CHECKPOINT}").write_text("")
    teacher.train(ws, ws, 1000, print)
    assert "--resume" not in calls[0] and calls[1][-2:] == ["--resume", "blueprint-1000.bin.ckpt"]


def test_the_teacher_plays_a_six_player_table_and_its_winrate_is_stored(ws, monkeypatch):
    from byosm.db.workspace import workspace_report
    from byosm.games.poker import table

    asked = []
    line = "winrate: +3390.1 mbb/hand (95% CI ±595.7) over 20000 hands in 5.0s"
    monkeypatch.setattr(table, "run", lambda ws, hands, args, log: asked.append(args) or ["loaded", line])
    won = table.play(ws, ws, 7, "caller", 20000, False, lambda _: None)
    assert won["mbb_per_hand"] == 3390.1 and won["ci95"] == 595.7 and won["hands"] == 20000 and won["seats"] == 6
    assert asked[0][asked[0].index("--players") + 1] == "6" and "--aivat" in asked[0] and "--search" not in asked[0]
    assert workspace_report(ws, table.key(7, "caller", False)) == won
    table.play(ws, ws, 7, "random", 10, True, lambda _: None)
    assert "--search" in asked[1]
    with pytest.raises(ValueError, match="unknown field"):
        table.play(ws, ws, 7, "sharks", 10, False, lambda _: None)
