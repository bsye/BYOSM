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


def test_a_feature_written_after_the_event_is_flagged():
    splits = build(_register(), _recipe(["kind", "noise", "status"]))
    flagged = {r["feature"] for r in check(splits["train"], ["kind", "status"]) if r["suspect"]}
    assert "status" in flagged and "noise" not in flagged


def test_the_model_beats_the_base_rate_on_a_later_cohort():
    out = evaluate(build(_register(), _recipe(["kind", "noise"])), ["kind"])
    assert out["gradient boosting"]["auc"] > out["base rate"]["auc"] + 0.1
    low, high = out["gradient boosting"]["auc_ci"]
    assert low <= out["gradient boosting"]["auc"] <= high
