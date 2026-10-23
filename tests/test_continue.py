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


class _Project:
    promoted = "v2"

    def __init__(self, checks, meta):
        self._meta = meta
        self.checks = checks

    def exp_spec(self, exp):
        return TaskSpec(
            name="q",
            description="q",
            modality="text",
            labels=[Label(name="a", definition=""), Label(name="b", definition="")],
        )

    def experiment(self, exp):
        return {"meta": self._meta, "errors": []}


def _recipe(monkeypatch, checks, meta):
    import byosm.train.plan as plan

    monkeypatch.setattr(plan, "_health", lambda p, parent: {c["id"]: c for c in checks})
    return plan.recipe(_Project(checks, meta), "v2")


def test_an_overfitting_model_is_continued_gently(monkeypatch):
    spec, reasons, _ = _recipe(monkeypatch, [{"id": "overfit", "status": "warn", "value": "warn"}], {})
    assert spec.train.epochs == 1 and "overfitting" in reasons[0]


def test_a_model_behind_its_target_and_still_improving_gets_more_epochs(monkeypatch):
    meta = {"history": [{}, {}, {}], "best_epoch": 2}
    spec, reasons, _ = _recipe(monkeypatch, [{"id": "target", "status": "warn", "value": "0.90 of 0.94"}], meta)
    assert spec.train.epochs > 3 and "still improving" in reasons[0]


def test_problems_training_cannot_fix_are_named_with_what_to_do_instead(monkeypatch):
    _, reasons, _ = _recipe(monkeypatch, [{"id": "speed", "status": "fail", "value": "30 of 10 ms"}], {})
    assert any("distill" in r for r in reasons)


def test_errors_on_the_weakest_label_come_first():
    from byosm.train.plan import _aimed

    errors = [{"gold": "x", "pred": "y"}, {"gold": "weak", "pred": "y"}, {"gold": "z", "pred": "weak"}]
    out = _aimed(errors, {"weakest": {"id": "weakest", "status": "warn", "value": "weak 0.79"}})
    assert [e["gold"] for e in out] == ["weak", "z", "x"]
