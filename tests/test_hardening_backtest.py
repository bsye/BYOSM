import numpy as np
import pandas as pd

from byosm.tabular.backtest import backtest, scorable
from byosm.tabular.snapshots import LABEL, Recipe


def _register(n: int = 1500) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    start = pd.Timestamp("1999-01-01") + pd.to_timedelta(rng.integers(0, 3650, n), unit="D")
    event = start + pd.to_timedelta(rng.integers(100, 6000, n), unit="D")
    event = event.where(rng.random(n) < 0.7)
    return pd.DataFrame({"start": start.astype(str), "event": event.astype(str), "size": rng.random(n)})


def _recipe() -> Recipe:
    return Recipe(start="start", event="event", horizon_years=1, cutoffs={}, features=["size"])


def test_scorable_rejects_empty_and_single_class_splits():
    both = pd.DataFrame({LABEL: [0, 1]})
    assert scorable({"train": both, "test": both})
    assert not scorable({"train": both.iloc[:0], "test": both})
    assert not scorable({"train": both, "test": pd.DataFrame({LABEL: [0, 0]})})


def test_backtest_skips_years_without_data_and_scores_the_rest():
    lines: list[str] = []
    out = backtest(_register(), _recipe(), [1990, 2006], [], [], lines.append)
    assert [x["year"] for x in out] == [2006]
    assert any("backtest 1990: skipped" in line for line in lines)
