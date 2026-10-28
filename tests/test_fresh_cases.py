import pytest

from byosm.evals.fresh import FRESH, finding, overfit_health
from byosm.health_score import UNKNOWN
from byosm.project import Project
from byosm.spec import EvalSuite, Label, TaskSpec


def _spec(task_type="classify"):
    labels = [Label(name=n, definition="") for n in "ab"]
    evals = [EvalSuite(name="builtin", primary=True), EvalSuite(name="llm-written", kind="generated")]
    return TaskSpec(name="tc", description="", modality="text", labels=labels, task_type=task_type, evals=evals)


@pytest.mark.parametrize(("new", "severity"), [(0.88, "ok"), (0.82, "warn"), (0.70, "fail")])
def test_a_drop_on_test_cases_written_after_training_is_graded(new, severity):
    results = {"builtin": {"objective": 0.9}, "llm-written": {"objective": new, "n_test": 400}}
    [f] = finding(_spec(), results)
    assert f["check"] == FRESH and f["severity"] == severity and f["suite"] == "llm-written"
    assert finding(_spec(), {"builtin": {"objective": 0.9}}) == []


def _project(spec):
    p = Project("tc").ensure()
    p.save_spec(spec)
    p.save_experiment("v1", meta={})
    p.promote("v1")
    return p


def test_the_overfit_check_stays_open_until_new_test_cases_are_written_and_scored(ws):
    from conftest import save_board

    spec = _spec()
    p = _project(spec)
    assert overfit_health(p, spec, None)["args"] == ["overfit", "tc"]
    report = {"experiment": "v1", "findings": [{"check": "gap"}], "verdict": "ok", "summary": "fine"}
    first = overfit_health(p, spec, report)
    assert first["status"] == UNKNOWN and first["args"] == ["testset", "tc"]
    save_board(p, {"v1": {"llm-written": {"kind": "experiment", "objective": 0.8}}})
    assert overfit_health(p, spec, report)["args"] == ["overfit", "tc"]
    done = report | {"findings": [{"check": "gap"}, {"check": FRESH}]}
    assert overfit_health(p, spec, done)["status"] == "ok"


def test_a_kind_that_cannot_write_test_sets_is_judged_without_them(ws):
    spec = _spec("frame")
    p = _project(spec)
    report = {"experiment": "v1", "findings": [{"check": "gap"}], "verdict": "warn", "summary": "some"}
    assert overfit_health(p, spec, report)["status"] == "warn"
