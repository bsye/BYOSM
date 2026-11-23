import csv

import numpy as np
import pytest
from typer.testing import CliRunner

from byosm import metrics, routing
from byosm.benchmarks import add as adding
from byosm.benchmarks import pokerbench
from byosm.benchmarks.catalog import fitting
from byosm.cli import app
from byosm.games.poker import create as creating
from byosm.games.poker.moves import LABELS, label_of, labelled
from byosm.games.poker.spots import postflop_spot, preflop_spot
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
    monkeypatch.setattr(adding, "licence", lambda dataset: "apache-2.0")


def test_pokerbench_fits_only_the_move_labels_and_its_floor_scores_the_label_share(tmp_path, monkeypatch):
    hub(tmp_path, monkeypatch)
    assert [b.name for b in fitting(spec())] == ["pokerbench"] and fitting(spec(families=False)) == []
    rows = pokerbench.rows(spec(), print)
    test = [r for r in rows if r["split"] == "test"]
    found = pokerbench.expected(spec(), rows)
    assert found["accuracy"] == sum(r["label"] == "fold" for r in test) / len(test)
    assert pokerbench.observed({"metrics": {"accuracy": 0.2, "family_accuracy": 0.3}}) == {
        "accuracy": 0.2,
        "family_accuracy": 0.3,
    }


def test_training_spots_are_balanced_by_label_and_never_copy_a_test_spot():
    pre, post = preflop_spot(PREFLOP), postflop_spot(POSTFLOP)
    other = postflop_spot({**POSTFLOP, "holding": "AhAd", "correct_decision": "Check"})
    picked = creating.balanced([pre, post, other, other], [pre], len(LABELS))
    assert pre not in picked and sorted(s.label for s in picked) == ["bet big", "check"]


def test_poker_new_builds_the_specialist_with_pokerbench_deciding_promotion(ws, monkeypatch):
    hub(ws, monkeypatch)
    result = CliRunner().invoke(app, ["poker", "new", "holdem"])
    assert result.exit_code == 0, result.output
    s = Project("holdem").spec()
    assert s.primary_suite() == "pokerbench" and s.objective.metric == "accuracy"
    assert s.label_families == [n.split()[0] for n in LABELS]
    assert "protocol check, accuracy" in result.output
    again = CliRunner().invoke(app, ["poker", "new", "holdem"])
    assert again.exit_code != 0 and "already exists" in again.output


def test_a_poker_request_in_home_creates_the_poker_specialist_and_others_go_on_as_before(ws, monkeypatch):
    hub(ws, monkeypatch)
    monkeypatch.setattr(routing, "route", lambda prompt, cfg: {"kind": routing.POKER})
    assert routing.routed("what should I do with KK facing a shove", print) == (routing.POKER, "poker-move")
    assert Project("poker-move").spec().description == "what should I do with KK facing a shove"
    monkeypatch.setattr(routing, "route", lambda prompt, cfg: {"kind": routing.SPECIALIST})
    assert routing.routed("is this email spam", print) == (routing.SPECIALIST, None)
