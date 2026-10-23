import numpy as np
import pandas as pd

from byosm.tabular.fit import evaluate
from byosm.tabular.leakage import check
from byosm.tabular.snapshots import AGE, LABEL, Recipe, build, snapshot


def _register(n=3000, seed=0):
    rng = np.random.default_rng(seed)
    start = pd.Timestamp("2000-01-01") + pd.to_timedelta(rng.integers(0, 365 * 12, n), unit="D")
    risky = rng.random(n) < 0.3
    life = np.where(risky, rng.integers(200, 2500, n), rng.integers(3000, 9000, n))
    end = start + pd.to_timedelta(life, unit="D")
    end = pd.Series(end).where(end < pd.Timestamp("2024-01-01"))
    return pd.DataFrame(
        {
            "start": start.strftime("%Y-%m-%d"),
            "end": end.dt.strftime("%Y-%m-%d"),
            "kind": np.where(risky, "a", "b"),
            "noise": rng.random(n),
            "status": np.where(end.isna() | (end >= pd.Timestamp("2016-01-01")), "renamed", "old"),
        }
    )


def _recipe(features):
    return Recipe("start", "end", 5, {"train": [2008, 2010], "test": [2013]}, features, per_cutoff=2000)


def test_snapshot_keeps_only_the_living_and_labels_the_horizon():
    df = pd.DataFrame({"start": ["2001-01-01", "2001-01-01", "2012-01-01"], "end": ["2009-06-01", None, None]})
    s = snapshot(df, Recipe("start", "end", 5, {}, []), 2010)
    assert len(s) == 1 and s[LABEL].tolist() == [0] and round(s[AGE].iloc[0]) == 9
    s = snapshot(df, Recipe("start", "end", 5, {}, []), 2008)
    assert sorted(s[LABEL].tolist()) == [0, 1]


def test_a_value_given_only_to_survivors_is_flagged_even_when_it_predicts_weakly():
    from byosm.tabular.leakage import pure_share

    rng = np.random.default_rng(1)
    y = pd.Series((rng.random(4000) < 0.25).astype(int))
    renamed = pd.Series(np.where((y == 0) & (rng.random(4000) < 0.5), "new", rng.choice(["a", "b"], 4000)))
    honest = pd.Series(rng.choice(["a", "b", "c"], 4000))
    assert pure_share(renamed, y) > 0.3 and pure_share(honest, y) == 0


def test_a_feature_written_after_the_event_is_flagged():
    splits = build(_register(), _recipe(["kind", "noise", "status"]))
    flagged = {r["feature"] for r in check(splits["train"], ["kind", "status"]) if r["suspect"]}
    assert "status" in flagged and "noise" not in flagged


def test_the_model_beats_the_base_rate_on_a_later_cohort():
    out = evaluate(build(_register(), _recipe(["kind", "noise"])), ["kind"])
    assert out["gradient boosting"]["auc"] > out["base rate"]["auc"] + 0.1
    low, high = out["gradient boosting"]["auc_ci"]
    assert low <= out["gradient boosting"]["auc"] <= high


def test_derived_columns_follow_their_steps():
    from byosm.tabular.derive import derive

    df = pd.DataFrame(
        {
            "addr": ["1 Main St, Cork", "2 Quay, cork", None],
            "code": ["45.21", "45.1", None],
            "kind": ["LTD guarantee", "LTD", "x"],
        }
    )
    out = derive(
        df,
        {
            "county": {"from": "addr", "steps": ["part", {"frequent": {"min_count": 2}}]},
            "major": {"from": "code", "steps": [{"part": {"sep": ".", "index": 0}}]},
            "family": {
                "from": "kind",
                "steps": [{"keywords": {"groups": {"GUARANTEE": "guarantee"}, "other": "shares"}}],
            },
        },
    )
    assert out["county"].tolist()[:2] == ["CORK", "CORK"] and out["county"].isna().iloc[2]
    assert out["major"].tolist()[:2] == ["45", "45"] and out["family"].tolist() == ["guarantee", "shares", "shares"]


def test_a_forecast_whose_train_labels_reach_the_test_years_is_refused(ws):
    import pytest

    from byosm.tabular.task import create, forecasts

    spec = {"title": "t", "source": "s", "start": "a", "event": "b", "horizon_years": 5, "features": ["x"]}
    with pytest.raises(ValueError, match="after the first test cutoff"):
        create(ws, "late", {**spec, "cutoffs": {"train": [2010], "test": [2012]}})
    create(ws, "ok", {**spec, "cutoffs": {"train": [2010], "test": [2015]}})
    assert [f["name"] for f in forecasts(ws)] == ["ok"]


def test_history_features_only_see_events_before_the_cutoff():
    from byosm.tabular.history import NEVER_DAYS, History, features

    events = pd.DataFrame({"id": ["a", "a", "b", "a"], "on": ["2023-03-01", "2021-06-01", "2023-12-31", "2024-02-01"]})
    keys = pd.Series(["a", "b", "c"])
    out = features(History("filings", events, "id", "on"), keys, pd.Timestamp("2024-01-01"))
    assert out["filings_last_year"].tolist() == [1, 1, 0]
    assert out["filings_days_since"].tolist() == [306, 1, NEVER_DAYS]


def test_text_columns_that_hold_numbers_become_numbers_and_the_rest_categories():
    from byosm.tabular.derive import typed

    df = pd.DataFrame({"size": ["3", "4.5", None], "kind": ["a", "b", "a"], "code": ["01", "02", "03"]})
    out, cats = typed(df, ["size", "kind", "code"], ["code"])
    assert cats == ["code", "kind"] and out["size"].tolist()[:2] == [3.0, 4.5]
    many = pd.DataFrame({"street": [f"s{i}" for i in range(400)]})
    capped, _ = typed(many, ["street"], [])
    assert capped["street"].nunique() == 251


def test_a_backtest_scores_each_test_year_on_windows_that_end_before_it():
    from byosm.tabular.backtest import backtest, windows

    assert windows(2016, 5) == {"train": [2007, 2009, 2011], "test": [2016]}
    r = Recipe("start", "end", 2, {}, ["kind", "noise"], per_cutoff=800)
    rows = backtest(_register(), r, [2016, 2014], ["kind"], [], log=lambda m: None)
    assert [x["year"] for x in rows] == [2014, 2016]
    assert all(x["scores"]["gradient boosting"]["auc"] > 0.6 for x in rows)


def test_a_test_year_whose_outcome_is_not_known_yet_is_refused(ws):
    import pytest

    from byosm.tabular.task import create

    spec = {
        "title": "t",
        "source": "s",
        "start": "a",
        "event": "b",
        "horizon_years": 5,
        "features": ["x"],
        "test_year": 2016,
    }
    assert create(ws, "fine", spec).cutoffs["train"] == [2007, 2009, 2011]
    with pytest.raises(ValueError, match="not known yet"):
        create(ws, "future", {**spec, "backtest": [2025]})


def test_today_scores_everyone_alive_now_with_training_years_moved_forward():
    from datetime import date

    from byosm.tabular.score import today

    df = _register(4000)
    r = Recipe("start", "end", 2, {"train": [2012], "test": [2016]}, ["kind", "noise"], per_cutoff=1500)
    now, everyone = today(df, r, ["kind"], [], ["kind"], log=lambda m: None)
    assert len(everyone) == now["scored"] and everyone["chance"].is_monotonic_decreasing
    assert now["trained_on"] == [2012 + date.today().year - 2016]
    chances = [t["p"] for t in now["top"]]
    assert now["scored"] == sum(b["n"] for b in now["bands"]) and chances == sorted(chances, reverse=True)


def test_the_reason_for_a_high_chance_is_the_feature_that_drives_it():
    from byosm.tabular.fit import _xy, models
    from byosm.tabular.reasons import reasons, typical

    splits = build(_register(), _recipe(["kind", "noise"]))
    x, y = _xy(splits["train"], ["kind"])
    model = models(["kind"])["gradient boosting"].fit(x, y)
    risky = x[x["kind"] == "a"].head(5)
    assert all(r and r[0]["feature"] == "kind" for r in reasons(model, risky, typical(x)))
