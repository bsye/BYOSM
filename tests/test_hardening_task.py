import pytest

from byosm.tabular.task import check_name, create, model_path, scored_path

SPEC = {
    "title": "t",
    "source": "s",
    "start": "a",
    "event": "b",
    "horizon_years": 5,
    "features": ["x"],
    "cutoffs": {"train": [2010], "test": [2015]},
}


def test_overwriting_a_forecast_drops_the_model_and_scores_of_the_old_spec(ws):
    create(ws, "churn", SPEC)
    scored_path(ws, "churn").parent.mkdir(parents=True, exist_ok=True)
    scored_path(ws, "churn").write_text("old")
    model_path(ws, "churn").write_bytes(b"old")
    create(ws, "churn", {**SPEC, "features": ["y"]})
    assert not scored_path(ws, "churn").exists()
    assert not model_path(ws, "churn").exists()


@pytest.mark.parametrize("name", ["", ".hidden", "../../tmp/x", "a/b", "a b", "a\\b"])
def test_a_forecast_name_that_could_leave_the_state_folder_is_refused(ws, name):
    with pytest.raises(ValueError):
        create(ws, name, SPEC)
    with pytest.raises(ValueError):
        model_path(ws, name)


def test_generated_style_names_are_accepted(ws):
    assert check_name("company-failure_5y") == "company-failure_5y"
    assert model_path(ws, "churn").name == "churn.joblib"
    assert scored_path(ws, "churn").parent == model_path(ws, "churn").parent
