import random

from byosm.improve.tpe import suggest, tpe_proposals
from byosm.project import Project
from byosm.spec import Label, TaskSpec

LR = [(1e-5, 0.70), (2e-5, 0.75), (3e-5, 0.90), (4e-5, 0.92), (1e-4, 0.60), (2e-4, 0.50), (5e-6, 0.55)]


def test_too_few_tried_values_give_no_suggestion():
    assert suggest(LR[:3], False, random.Random(0)) is None


def test_the_suggestion_lands_near_the_values_that_scored_best():
    picks = [suggest(LR, False, random.Random(i)) for i in range(20)]
    assert all(1.2e-5 <= v <= 1e-4 for v in picks)
    assert sum(2e-5 <= v <= 6e-5 for v in picks) >= 15


def test_an_integer_setting_gets_a_whole_number_of_at_least_one():
    epochs = [(2, 0.6), (4, 0.8), (6, 0.85), (12, 0.7), (16, 0.65)]
    v = suggest(epochs, True, random.Random(0))
    assert isinstance(v, int) and v >= 1


def test_tpe_proposes_a_value_once_a_setting_has_enough_scored_tries(ws):
    from conftest import save_board

    p = Project("tc").ensure()
    spec = TaskSpec(name="tc", description="", modality="text", labels=[Label(name=n, definition="") for n in "ab"])
    p.save_spec(spec)
    for i, (lr, calib) in enumerate(LR[:5]):
        edited = spec.model_copy(deep=True)
        edited.train.lr_backbone = lr
        op = {"op": "set", "path": "train.lr_backbone", "value": lr}
        p.save_experiment(f"v{i + 1}", spec=edited, meta={}, proposal={"applied": [{"id": "x", "ops": [op]}]})
        save_board(p, {f"v{i + 1}": {"builtin": {"kind": "experiment", "objective": calib, "calib_objective": calib}}})
    found = tpe_proposals(p, spec)
    assert [pr["id"] for pr in found] == ["tpe-train.lr_backbone"]
    assert found[0]["source"] == "tpe" and "5 tried values" in found[0]["why"]
