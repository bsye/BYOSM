import pytest
from conftest import paired, save_board

from byosm.project import Project
from byosm.promotion import maybe_promote
from byosm.spec import EvalSuite, Label, TaskSpec

BOTH = ["builtin", "hard"]


def _spec(selection: list[str]) -> TaskSpec:
    labels = [Label(name="a", definition=""), Label(name="b", definition="")]
    evals = [EvalSuite(name="builtin", primary=True), EvalSuite(name="hard", kind="hf")]
    return TaskSpec(
        name="tc",
        description="",
        modality="text",
        labels=labels,
        finalist_seeds=1,
        evals=evals,
        selection_suites=selection,
    )


def _scored(calib: float, n_calib: int = 200) -> dict:
    return {
        "kind": "experiment",
        "n_test": 500,
        "n_calib": n_calib,
        "objective": calib,
        "calib_objective": calib,
    } | paired(calib)


def _vote(selection: list[str], v1: tuple[float, float], v2: tuple[float, float], hard_calib: int = 200) -> tuple:
    p = Project("tc").ensure()
    p.save_spec(_spec(selection))
    for e in ("v1", "v2"):
        p.save_experiment(e, meta={"calib_held": True})
    p.promote("v1", "test")
    for e, (builtin, hard) in (("v1", v1), ("v2", v2)):
        save_board(p, {e: {"builtin": _scored(builtin), "hard": _scored(hard, hard_calib)}})
    said = []
    maybe_promote(p, "v2", said.append)
    return p.promoted, said[-1]


def test_a_win_on_the_primary_suite_that_loses_the_mean_keeps_the_promoted_model(ws):
    promoted, said = _vote(BOTH, v1=(0.80, 0.80), v2=(0.90, 0.60))
    assert promoted == "v1" and said.startswith("  kept v1: calib mean over builtin, hard 0.8000 >= v2 0.7500")
    assert "builtin +0.1000, hard -0.2000" in said


def test_a_sure_lead_on_the_mean_promotes_and_names_each_suite(ws):
    promoted, said = _vote(BOTH, v1=(0.70, 0.70), v2=(0.95, 0.90))
    assert promoted == "v2" and "calib mean over builtin, hard macro_f1 0.9250 (was 0.7000" in said
    assert "builtin +0.2500, hard +0.2000" in said


def test_without_selection_suites_only_the_primary_suite_decides(ws):
    promoted, said = _vote([], v1=(0.70, 0.90), v2=(0.95, 0.50))
    assert promoted == "v2" and "builtin calib macro_f1 0.9500" in said


def test_a_selection_suite_with_too_few_calib_items_is_left_out(ws):
    promoted, said = _vote(BOTH, v1=(0.70, 0.90), v2=(0.95, 0.50), hard_calib=10)
    assert promoted == "v2" and "builtin calib" in said and "hard" not in said


def test_a_selection_suite_must_be_one_of_the_specs_suites():
    with pytest.raises(ValueError, match="not among the spec's evals"):
        _spec(["builtin", "missing"])
