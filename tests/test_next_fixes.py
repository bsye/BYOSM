from types import SimpleNamespace

from byosm.next_fixes import less_overfit
from byosm.spec import Label, TaskSpec


def test_an_overfit_model_gets_an_experiment_with_a_lower_rate_stopping_at_the_best_epoch():
    spec = TaskSpec(name="q", description="q", modality="text", labels=[Label(name="a", definition="")])
    report = {
        "findings": [
            {"check": "gap", "severity": "fail", "numbers": {"train": 0.956, "test": 0.788}},
            {"check": "curve", "severity": "warn", "numbers": {"best_epoch": 4}},
        ]
    }
    p = SimpleNamespace(name="q", promoted="v1", report=lambda kind: report, experiment=lambda e: None)
    step = less_overfit(p, spec, [{"id": "overfit", "status": "fail"}])
    assert "0.956" in step["why"] and "epoch 4" in step["why"]
    assert "train.epochs=4" in step["args"] and step["args"][:4] == ["exp", "create", "q", "fit-v1"]
    assert less_overfit(p, spec, [{"id": "overfit", "status": "ok"}]) is None
