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


def _project(findings, experiment=None):
    return SimpleNamespace(
        name="q", promoted="v1", report=lambda kind: {"findings": findings}, experiment=lambda e: experiment
    )


def _spec():
    return TaskSpec(name="q", description="q", modality="text", labels=[Label(name="a", definition="")])


FAILING = [{"id": "overfit", "status": "warn"}]


def test_a_model_whose_dev_was_still_rising_trains_longer_instead_of_slower():
    rising = {"check": "curve", "severity": "warn", "numbers": {"best_epoch": 3, "epochs_run": 3}}
    step = less_overfit(_project([rising]), _spec(), FAILING)
    assert step["id"] == "train-longer" and f"train.epochs={_spec().train.epochs * 2}" in step["args"]


def test_drift_between_calib_and_test_alone_is_not_treated_as_overfitting():
    drift = {"check": "drift", "severity": "warn", "numbers": {}}
    assert less_overfit(_project([drift]), _spec(), FAILING) is None


def test_a_fix_whose_build_never_finished_is_offered_again_as_a_build():
    gap = {"check": "gap", "severity": "fail", "numbers": {"train": 0.95, "test": 0.7}}
    step = less_overfit(_project([gap], experiment={"meta": None}), _spec(), FAILING)
    assert step["args"] == ["build", "q", "--exp", "fit-v1"]
    assert less_overfit(_project([gap], experiment={"meta": {"dev": 1}}), _spec(), FAILING) is None
