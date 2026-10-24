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
    model = Boosted()
    model.fit(histories, 24, 48)
    batch = model.predict_many([t["history"] for t in tests], 48)
    assert all(np.allclose(b, model.predict(t["history"], 48)) for b, t in zip(batch, tests, strict=True))


@pytest.mark.slow
def test_a_series_that_is_zero_half_the_time_is_not_forecast_as_all_zeros():
    t = np.arange(24 * 120)
    sun = [np.clip(np.sin(2 * np.pi * (t - 6) / 24), 0, None) * (100 + i) for i in range(4)]
    series = [Series(str(i), "H", v) for i, v in enumerate(sun)]
    cut = windows(series) * 48
    model = Boosted()
    model.fit([s.values[: len(s.values) - cut] for s in series], 24, 48)
    median = model.predict(series[0].values[: len(t) - cut], 48)[QUANTILES.index(0.5)]
    assert median.max() > 50


def test_the_patch_model_returns_sorted_quantiles_for_every_window(monkeypatch):
    import byosm.series.patch as patch

    monkeypatch.setattr(patch, "STEPS", 5)
    monkeypatch.setattr(patch, "BATCH", 8)
    histories = [_wave(seed=i) for i in range(3)]
    model = patch.PatchModel()
    model.fit(histories, 24, 48)
    out = model.predict_many([h[:-48] for h in histories] + [np.full(10, np.nan)], 48)
    assert len(out) == 4 and all(o.shape == (len(QUANTILES), 48) for o in out)
    assert all(np.all(np.diff(o, axis=0) >= 0) for o in out)
