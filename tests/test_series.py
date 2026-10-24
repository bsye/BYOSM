import numpy as np
import pytest

from byosm.series.gifteval import Series, horizon, season, splits, unit, windows
from byosm.series.metrics import QUANTILES, crps, geomean, mase
from byosm.series.models import Boosted, SeasonalNaive


def _wave(n=24 * 60, noise=0.1, seed=0):
    t = np.arange(n)
    return 10 + np.sin(2 * np.pi * t / 24) + np.random.default_rng(seed).normal(0, noise, n)


def test_frequencies_map_to_the_gifteval_horizon_and_season():
    assert (unit("H"), horizon("H"), season("H")) == ("H", 48, 24)
    assert (unit("10S"), horizon("15T"), season("D")) == ("S", 48, 7)


def test_test_windows_sit_at_the_end_and_never_overlap_the_training_history():
    series = [Series("a", "H", _wave())]
    tests = splits(series)
    assert len(tests) == windows(series) and all(len(t["future"]) == 48 for t in tests)
    assert len(tests[-1]["history"]) + 48 == len(series[0].values)


def test_a_perfect_forecast_scores_zero_and_mase_scales_by_the_seasonal_naive_error():
    history, future = _wave(240), _wave(288)[240:]
    exact = np.repeat(future[None], len(QUANTILES), axis=0)
    assert mase(history, future, future, 24) == 0 and crps(future, exact)[0] == 0
    assert geomean([0.5, 2.0]) == 1.0


@pytest.mark.slow
def test_the_boosted_specialist_beats_seasonal_naive_on_a_seasonal_series():
    series = [Series(str(i), "H", _wave(seed=i)) for i in range(4)]
    tests = splits(series)
    cut = windows(series) * 48
    histories = [s.values[: len(s.values) - cut] for s in series]
    errors = {}
    for model in (SeasonalNaive(), Boosted()):
        model.fit(histories, 24, 48)
        errors[model.name] = np.mean(
            [mase(t["history"], t["future"], model.predict(t["history"], 48)[QUANTILES.index(0.5)], 24) for t in tests]
        )
    assert errors["gradient boosting"] < errors["seasonal naive"]
