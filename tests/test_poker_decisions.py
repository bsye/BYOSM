from __future__ import annotations

import json

import pytest

from byosm.games.poker import sharingan
from byosm.games.poker.phh import parse
from byosm.games.poker.sharingan import compare, decisions

HAND = """variant = 'NT'
antes = [0, 0, 0, 0, 0, 0]
blinds_or_straddles = [50, 100, 0, 0, 0, 0]
min_bet = 100
starting_stacks = [10000, 10000, 10000, 10000, 10000, 10000]
actions = ['d dh p1 TcQc', 'd dh p2 ????', 'd dh p3 9c3d', 'd dh p4 Ah4h', 'd dh p5 ????', 'd dh p6 6c7s', 'p3 f', 'p4 cbr 210', 'p5 f', 'p6 f', 'p1 cc', 'p2 f', 'd db 7d5h9d', 'p1 cc', 'p4 cc', 'd db 7c', 'p1 cc', 'p4 cc', 'd db Qh', 'p1 cbr 230', 'p4 f', 'p5 sm KsKd']
players = ['MrBlue', 'MrBlonde', 'MrWhite', 'MrPink', 'MrBrown', 'Pluribus']
"""


def test_a_logged_hand_becomes_one_request_per_decision_whose_cards_are_known():
    found = decisions.asked("100/0", parse(HAND))
    assert [(a.player, a.street, a.kind) for a in found] == [
        ("MrWhite", "preflop", "fold"),
        ("MrPink", "preflop", "raise"),
        ("MrBrown", "preflop", "fold"),
        ("Pluribus", "preflop", "fold"),
        ("MrBlue", "preflop", "call"),
        ("MrBlue", "flop", "check"),
        ("MrPink", "flop", "check"),
        ("MrBlue", "turn", "check"),
        ("MrPink", "turn", "check"),
        ("MrBlue", "river", "raise"),
        ("MrPink", "river", "fold"),
    ]
    first, shown, river = found[0].request, found[2].request, found[-2].request
    assert first["hero_seat"] == 2 and first["hole_cards"] == ["9c", "3d"] and first["history"] == []
    assert first["button_seat"] == 5 and first["players"][0] == {"seat": 0, "stack_start": 10000}
    assert first["options"] == {"sample": False, "search": False} and first["hand_id"] == "100/0:6"
    assert shown["hole_cards"] == ["Ks", "Kd"]
    assert shown["history"] == [{"seat": 2, "type": "fold"}, {"seat": 3, "type": "raise", "to": 210}]
    assert river["board"] == ["7d", "5h", "9d", "7c", "Qh"]
    assert river["history"][-2:] == [{"seat": 0, "type": "check"}, {"seat": 3, "type": "check"}]
    assert found[-1].request["history"][-1] == {"seat": 0, "type": "bet", "to": 230}


def test_chips_are_scaled_so_the_big_blind_is_one_hundred():
    small = HAND.replace("[50, 100, 0, 0, 0, 0]", "[0.1, 0.25, 0, 0, 0, 0]").replace("10000", "25")
    small = small.replace("cbr 210", "cbr 0.75").replace("cbr 230", "cbr 1.5")
    found = decisions.asked("h", parse(small))
    assert found[0].request["small_blind"] == 40 and found[0].request["big_blind"] == 100
    assert found[0].request["players"][0]["stack_start"] == 10000
    assert found[2].request["history"][-1] == {"seat": 3, "type": "raise", "to": 300}


@pytest.mark.parametrize(
    ("change", "why"),
    [
        (("variant = 'NT'", "variant = 'PO'"), "not no-limit"),
        (("antes = [0, 0, 0, 0, 0, 0]", "antes = [5, 5, 5, 5, 5, 5]"), "antes"),
        (("[50, 100, 0, 0, 0, 0]", "[50, 100, 200, 0, 0, 0]"), "blinds"),
    ],
)
def test_hands_the_blueprint_was_not_trained_for_are_refused(change, why):
    assert decisions.refused(parse(HAND)) is None
    assert why in decisions.refused(parse(HAND.replace(*change)))
    assert "players" in decisions.refused({**parse(HAND), "players": ["a", "b"], "blinds_or_straddles": [50, 100]})


def test_a_move_is_judged_by_its_kind_and_the_weight_the_blueprint_gave_it():
    logged = decisions.asked("100/0", parse(HAND))[1]
    probs = [{"type": "fold", "p": 0.6}, {"type": "raise", "to": 250, "p": 0.3}, {"type": "all_in", "p": 0.1}]
    wrong = decisions.judged(logged, {"action": {"type": "fold"}, "probs": probs, "fallback_reason": None})
    right = decisions.judged(logged, {"action": {"type": "bet"}, "probs": probs, "fallback_reason": "off_tree"})
    assert wrong == {"same": False, "weight": pytest.approx(0.4), "fell_back": False}
    assert right["same"] and right["fell_back"]
    assert decisions.pooled([wrong, right]) == {
        "decisions": 2,
        "same_kind": 0.5,
        "mean_weight": pytest.approx(0.4),
        "fell_back": 0.5,
    }
    assert decisions.pooled([]) == {"decisions": 0}


def test_the_comparison_runs_sealed_and_groups_the_moves_by_who_made_them(ws, monkeypatch):
    from byosm.db.workspace import workspace_reports

    root = sharingan.work_dir(ws)
    (root / sharingan.WEIGHTS).mkdir()
    (root / sharingan.WEIGHTS / "sharingan-2.plcp").write_bytes(b"")
    (root / sharingan.DATA / sharingan.BUCKETS).mkdir(parents=True)
    ran = []

    def runner(command, turn, who, args, log):
        ran.append(command)
        asked = [json.loads(line) for line in (root / "work" / "pluribus.requests.jsonl").read_text().splitlines()]
        said = [
            {
                "key": a["key"],
                "ok": True,
                "response": {"action": {"type": "fold"}, "probs": [{"type": "fold", "p": 1.0}]},
            }
            for a in asked[:-1]
        ]
        said.append({"key": asked[-1]["key"], "ok": False, "error": {"code": "out_of_turn"}})
        (root / "work" / "pluribus.answers.jsonl").write_text("".join(json.dumps(s) + "\n" for s in said))
        return []

    monkeypatch.setattr(compare, "streamed", runner)
    monkeypatch.setattr(compare, "ensure_image", lambda log: None)
    monkeypatch.setattr("byosm.games.poker.sharingan.arena.limits", lambda: ("4", "8g"))
    hands = [("100/0", parse(HAND)), ("100/1", {**parse(HAND), "antes": [5] * 6})]
    made = compare.compared(ws, "sharingan-2", "pluribus", hands, compare.against_pluribus, lambda _: None)
    assert made["hands"] == 2 and made["asked"] == 11 and made["skipped"] == {"antes": 1}
    assert made["hands_showing_cards"] == 1 and made["decisions_with_cards"] == 11
    assert made["errors"] == {"out_of_turn": 1}
    assert made["groups"]["Pluribus"] == {"decisions": 1, "same_kind": 1.0, "mean_weight": 1.0, "fell_back": 0.0}
    assert made["groups"]["professionals"]["decisions"] == 9 and made["groups"]["all"]["decisions"] == 10
    assert made["groups"]["street:preflop"]["same_kind"] == pytest.approx(3 / 5)
    sent = ran[0]
    assert "none" in sent and f"{root / 'work'}:/work" in sent and f"{root / sharingan.WEIGHTS}:/weights:ro" in sent
    assert sent[sent.index("python3") :][:3] == ["python3", "-I", "/asker/asker.py"]
    assert workspace_reports(ws, "poker-decisions:")["poker-decisions:sharingan-2:pluribus"]["asked"] == 11


SET = """[1]
variant = 'NT'
antes = [0, 0, 0]
blinds_or_straddles = [0.25, 0.50, 0]
min_bet = 0.50
starting_stacks = [50.00, 25.00, 55.00]
actions = ['d dh p1 ????', 'd dh p2 ????', 'd dh p3 ????', 'p3 cbr 1.50', 'p1 f', 'p2 cc', 'd db JdJcQh', 'p2 cc', 'p3 cc', 'd db 9d', 'p2 cc', 'p3 cc', 'd db 3d', 'p2 cc', 'p3 cc', 'p2 sm Qc8c', 'p3 sm Qd9s']
venue = 'Somewhere'
time = 20:05:48
players = ['a', 'b', 'c']

[2]
variant = 'NT'
antes = [0, 0, 0]
blinds_or_straddles = [0.25, 0.50, 0]
min_bet = 0.50
starting_stacks = [50.00, 50.00, 50.00]
actions = ['d dh p1 ????', 'd dh p2 ????', 'd dh p3 ????', 'p3 f', 'p1 f']
players = ['a', 'b', 'c']
"""


def test_real_hands_are_read_from_the_archive_by_site_and_stake_and_only_shown_cards_are_asked(ws, monkeypatch):
    import zipfile

    from byosm.games.poker import real_hands

    with pytest.raises(ValueError, match="real-hands-fetch"):
        list(real_hands.hands(ws, 1.0))
    path = real_hands.archive(ws)
    path.parent.mkdir(parents=True)
    entry = "data/handhq/PS-2009-07-01_2009-07-23_50NLH_OBFU/0.5/ps NLH handhq_1-OBFUSCATED.phhs"
    with zipfile.ZipFile(path, "w") as made:
        made.writestr(entry, SET)
        made.writestr("data/pluribus/100/0.phh", HAND)
    assert real_hands.named(entry) == ("PS", "50")
    found = list(real_hands.hands(ws, 1.0))
    assert [hand_id for hand_id, _ in found] == [
        "PS-50/ps NLH handhq_1-OBFUSCATED#1",
        "PS-50/ps NLH handhq_1-OBFUSCATED#2",
    ]
    assert list(real_hands.hands(ws, 0.0)) == []
    asked, counts, skipped = compare.collected(found, compare.full_stack)
    assert counts == {"hands": 2, "hands_showing_cards": 1, "decisions_with_cards": 8} and not skipped
    assert [a.player for a in asked] == ["c", "c", "c", "c"]
    assert compare.by_stake(asked[0]) == ["stake:00050", "site:PS"]
    assert asked[0].request["players"][2]["stack_start"] == 11000
    with pytest.raises(ValueError, match="not the published"):
        real_hands.fetched(ws, lambda _: None)
