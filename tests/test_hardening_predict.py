from datetime import date

import pytest
from test_tabular import _register

from byosm.tabular.snapshots import Recipe

SPEC = {
    "title": "t",
    "source": "s",
    "start": "start",
    "event": "end",
    "horizon_years": 2,
    "features": ["kind", "noise"],
    "cutoffs": {"train": [2012], "test": [2016]},
}


@pytest.fixture
def forecast(ws, monkeypatch):
    import joblib

    import byosm.tabular.task as task
    from byosm.tabular.score import today

    df = _register(3000)
    r = Recipe("start", "end", 2, SPEC["cutoffs"], SPEC["features"], per_cutoff=1500)
    fitted = today(df, r, ["kind"], [], ["kind"], log=lambda m: None)[2]
    task.create(ws, "f", SPEC)
    task.model_path(ws, "f").parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(fitted, task.model_path(ws, "f"))
    monkeypatch.setattr(task, "histories", lambda spec, log: [])
    return ws


def _predict(ws, row, usual=()):
    from byosm.tabular.predict import predict

    return predict(ws, "f", row, log=lambda m: None, usual=usual)


def test_a_row_holding_only_start_and_one_feature_is_scored(forecast):
    out = _predict(forecast, {"start": "2010-01-01", "kind": "a"})
    assert out["state"] == "alive" and 0 <= out["chance"] <= 1


def test_a_cleared_field_counts_as_missing(forecast):
    out = _predict(forecast, {"start": "2010-01-01", "kind": "a", "noise": "", "end": ""})
    assert out["state"] == "alive" and 0 <= out["chance"] <= 1


def test_an_entity_that_started_this_year_or_has_no_start_is_not_scored_rather_than_ended(forecast):
    this_year = f"{date.today().year}-01-02"
    assert _predict(forecast, {"start": this_year, "kind": "a"})["state"] == "unscored"
    assert _predict(forecast, {"start": "", "kind": "a"})["state"] == "unscored"
    assert _predict(forecast, {"start": "not a date", "kind": "a"})["state"] == "unscored"


def test_only_an_event_in_the_past_is_ended(forecast):
    assert _predict(forecast, {"start": "2010-01-01", "end": "2020-01-01", "kind": "a"})["state"] == "ended"


def test_every_feature_has_a_signed_effect_and_setting_one_to_typical_moves_the_chance_by_it(forecast):
    row = {"start": "2010-01-01", "kind": "a", "noise": 0.3}
    out = _predict(forecast, row)
    drops = [abs(e["drop"]) for e in out["effects"]]
    assert {e["feature"] for e in out["effects"]} >= {"kind", "noise"} and drops == sorted(drops, reverse=True)
    top = out["effects"][0]
    assert {"value", "typical"} <= top.keys()
    moved = _predict(forecast, row, usual=[top["feature"]])
    assert moved["usual"] == [top["feature"]] and abs(out["chance"] - moved["chance"] - top["drop"]) < 1e-3
    assert next(e for e in moved["effects"] if e["feature"] == top["feature"])["drop"] == 0


def test_setting_an_unknown_feature_to_typical_fails_loudly(forecast):
    with pytest.raises(ValueError):
        _predict(forecast, {"start": "2010-01-01", "kind": "a"}, usual=["invented"])
