import numpy as np
from conftest import Flat

from byosm.series.gifteval import Series
from byosm.series.metrics import QUANTILES
from byosm.series.models import Ensemble, SeasonalNaive
from byosm.series.multirate import MultiRate, coarse, native, twin_name
from byosm.series.prediction import MULTI_RATES, PredictCfg

MEDIAN = QUANTILES.index(0.5)


class RightAtHalfRate:
    name = "right at half rate"

    def predict_many(self, histories, season, horizon, clocks=None):
        return [np.full((len(QUANTILES), horizon), h[-1] + (0.0 if season == 12 else 5.0)) for h in histories]


def _wave(n, seed=0):
    t = np.arange(n)
    return 10 + np.sin(2 * np.pi * t / 24) + np.random.default_rng(seed).normal(0, 0.3, n)


def test_a_coarse_series_averages_blocks_that_end_at_the_forecast_origin_and_skips_missing_values():
    history = np.arange(10.0)
    assert coarse(history, 4).tolist() == [3.5, 7.5]
    history[9] = np.nan
    assert coarse(history, 4).tolist() == [3.5, 7.0]
    history[6:] = np.nan
    assert np.isnan(coarse(history, 4)[-1]) and coarse(history, 1) is history


def test_a_coarse_forecast_is_interpolated_back_to_every_native_step_of_the_horizon():
    back = native(np.array([[0.0, 4.0], [2.0, 2.0]]), 2, 4)
    assert back.shape == (2, 4)
    assert back[0].tolist() == [0.0, 1.0, 3.0, 4.0] and back[1].tolist() == [2.0] * 4


def test_the_mix_trusts_the_rate_that_forecast_the_series_own_last_windows_best():
    flat = np.full(400, 10.0)
    plain = RightAtHalfRate().predict_many([flat], 24, 48)[0]
    mixed = MultiRate(RightAtHalfRate(), MULTI_RATES).predict_many([flat], 24, 48)[0]
    assert np.allclose(plain, 15.0) and np.allclose(mixed, 10.0, atol=1e-6)


class WorseAtCoarseRates:
    name = "worse at coarse rates"
    misses = {24: 1.0, 12: 1.5, 6: 3.0}

    def predict_many(self, histories, season, horizon, clocks=None):
        return [np.full((len(QUANTILES), horizon), h[-1] + self.misses[season]) for h in histories]


def test_a_rate_whose_recent_loss_is_half_again_the_best_gets_almost_no_weight():
    mixed = MultiRate(WorseAtCoarseRates(), MULTI_RATES).predict_many([np.full(400, 10.0)], 24, 48)[0]
    assert np.allclose(mixed, 11.0, atol=0.02)


def test_a_series_too_short_for_the_coarsest_rate_keeps_the_native_forecast_and_every_mix_is_sorted():
    short, long = _wave(100), _wave(24 * 30, seed=1)
    out = MultiRate(SeasonalNaive(), MULTI_RATES).predict_many([short, long], 24, 48)
    assert all(o.shape == (len(QUANTILES), 48) for o in out)
    assert np.array_equal(out[0], SeasonalNaive().predict_many([short], 24, 48)[0])
    assert all(np.all(np.diff(o, axis=0) >= 0) for o in out)


def test_with_multi_rate_on_every_model_and_the_ensemble_get_a_twin_scored_on_the_same_windows(monkeypatch):
    import byosm.series.fit as fit

    series = [Series(f"s{i}", "H", _wave(400, seed=i)) for i in range(3)]
    monkeypatch.setattr("byosm.series.scoring.load", lambda config: series)
    monkeypatch.setattr(fit, "emit", lambda kind, **data: None)
    ours = ["gradient boosting", "patch transformer"]
    monkeypatch.setattr(fit, "candidates", lambda effort: [SeasonalNaive(), Flat(ours[0], 1.0), Flat(ours[1], -1.0)])
    cfg = PredictCfg(rates=MULTI_RATES)
    per, trained = fit.fit_domain(["a"], 0, lambda _: None, lambda per, pending: None, cfg)
    twins = [twin_name(n) for n in [*ours, Ensemble.name]]
    assert set(twins) <= set(per["a"]) and set(twins) <= set(trained)
    assert all(isinstance(m, MultiRate) for m in trained[twins[2]].models)
    assert per["a"][twins[0]]["train_s"] == per["a"][ours[0]]["train_s"]
    assert per["a"][twins[0]]["calib_mase"] != per["a"][ours[0]]["calib_mase"]


def test_each_version_records_its_prediction_options_and_the_next_training_starts_from_the_promoted_ones(
    ws, monkeypatch
):
    import byosm.series.run as run
    from byosm.series.board import reports, save_report
    from byosm.series.specialist import NAIVE, PREDICT

    asked = []
    naive = {"mase": 1.0, "crps": 1.0, "calib_mase": 1.0, "calib_crps": 1.0, "windows": [1.0]}
    ours = {**naive, "mase": 0.8, "calib_mase": 0.8, "windows": [0.8]}

    def fit(configs, effort, log, progress, cfg):
        asked.append(cfg)
        return {"a": {NAIVE: naive, "gradient boosting": ours}}, {"gradient boosting": SeasonalNaive()}

    monkeypatch.setattr(run, "DOMAINS", {"demo": ["a"]})
    monkeypatch.setattr(run, "fit_domain", fit)
    monkeypatch.setattr("byosm.series.published.scores", lambda config, naive_mase: {})
    first = run.run_domain(ws, "demo", log=lambda _: None)
    assert first["versions"][0][PREDICT] == PredictCfg().model_dump()
    first["versions"][0][PREDICT] = PredictCfg(rates=MULTI_RATES).model_dump()
    save_report(ws, first)
    run.run_domain(ws, "demo", log=lambda _: None)
    assert asked == [PredictCfg(), PredictCfg(rates=MULTI_RATES)]
    assert next(r for r in reports(ws) if r["domain"] == "demo")["versions"][1][PREDICT]["rates"] == MULTI_RATES
