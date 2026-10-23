import pytest
import torch

from byosm.spec import Label, TaskSpec
from byosm.train.warm import CONTINUE_LR_FACTOR, _rows_by_label, continued


def test_a_head_row_follows_its_label_and_a_new_label_starts_fresh():
    old = torch.tensor([[1.0, 1.0], [2.0, 2.0]])
    fresh = torch.zeros(3, 2)
    out = _rows_by_label(old, fresh, ["a", "b"], ["b", "new", "a"])
    assert out.tolist() == [[2.0, 2.0], [0.0, 0.0], [1.0, 1.0]]
    assert _rows_by_label(old, torch.zeros(3, 5), ["a", "b"], ["a", "b", "c"]) is None


def test_continuing_lowers_the_learning_rate_and_trains_for_fewer_epochs():
    spec = TaskSpec(name="q", description="q", modality="text", labels=[Label(name="a", definition="")])
    out = continued(spec, "v4")
    assert out.train.init_from == "v4" and out.train.epochs <= 3
    assert out.train.lr_backbone == pytest.approx(spec.train.lr_backbone * CONTINUE_LR_FACTOR)
    assert spec.train.init_from is None


@pytest.mark.slow
def test_a_continued_experiment_starts_from_its_parent_weights(ws, capfd):
    from byosm.pipeline import build, next_exp
    from byosm.project import Project
    from byosm.spec import compile_spec

    spec = compile_spec("isq", "detect questions")
    spec.data.dev_size, spec.data.per_call = 40, 20
    spec.train.epochs, spec.train.pretrain_steps, spec.train.refine_rounds = 2, 0, 0
    p = Project("isq").ensure()
    p.save_spec(spec)
    build("isq", log=lambda m: None)
    exp = next_exp(p)
    p.save_experiment(exp, spec=continued(p.exp_spec("v1"), "v1", epochs=1))
    capfd.readouterr()
    build("isq", exp, log=print)
    assert "continuing from v1" in capfd.readouterr().out
    assert p.board()[exp]["builtin"]["objective"] > 0.8
