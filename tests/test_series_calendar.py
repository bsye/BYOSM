import datetime as dt

import numpy as np
import pytest

from byosm.series.calendar import FIELDS, UNKNOWN, clock, features
from byosm.series.gifteval import Series, splits
from byosm.series.metrics import QUANTILES
from byosm.series.models import Group, QuantileBoosted
from byosm.series.multirate import MultiRate

HOUR, WEEKDAY, MINUTE, MONTH = (FIELDS.index(f) for f in ("hour", "dayofweek", "minute", "month"))
MONDAY_10PM = dt.datetime(2024, 1, 1, 22)


class Recorder:
    name = "recorder"

    def __init__(self):
        self.seen = []

    def predict_many(self, histories, season, horizon, clocks=None):
        self.seen.append((season, clocks))
        return [np.zeros((len(QUANTILES), horizon)) for _ in histories]


def test_a_clock_gives_the_calendar_of_every_step_at_each_frequency():
    hourly = features(clock(MONDAY_10PM, "H"), 3)
    assert hourly[:, HOUR].tolist() == [22, 23, 0] and hourly[:, WEEKDAY].tolist() == [0, 0, 1]
    assert features(clock(MONDAY_10PM, "15T"), 3)[:, MINUTE].tolist() == [0, 15, 30]
    assert features(clock(MONDAY_10PM, "M"), 3)[:, MONTH].tolist() == [1, 2, 3]
    assert features(clock(MONDAY_10PM, "W-SUN"), 2)[:, WEEKDAY].tolist() == [0, 0]
    assert clock(None, "H") is None and (features(None, 2) == UNKNOWN).all()


def test_windows_and_prepared_datasets_carry_the_series_start_to_every_forecast(monkeypatch):
    from byosm.series.scoring import Scored, prepare, score_model

    series = [Series("a", "H", np.arange(24 * 40.0), MONDAY_10PM)]
    assert all(w["start"] == MONDAY_10PM for w in splits(series))
    monkeypatch.setattr("byosm.series.scoring.load", lambda config: series)
    p, model = prepare("a", lambda _: None), Recorder()
    score_model(model, Scored([p]), 0.0, lambda _: None)
    assert p.clocks == [clock(MONDAY_10PM, "H")] and all(set(c) == {clock(MONDAY_10PM, "H")} for _, c in model.seen)


def test_multi_rate_forecasts_keep_the_dates_at_the_native_rate_only():
    model = Recorder()
    MultiRate(model, [1, 2]).predict_many([np.ones(400)], 24, 48, [clock(MONDAY_10PM, "H")])
    assert {season: c is not None for season, c in model.seen} == {24: True, 12: False}


def _weekdays(start, n=24 * 7 * 12, seed=0):
    stamps = features(clock(start, "H"), n)
    weekend = stamps[:, WEEKDAY] >= 5
    t = np.arange(n)
    return 20 + 3 * np.sin(2 * np.pi * t / 24) - 8 * weekend + np.random.default_rng(seed).normal(0, 0.5, n)


@pytest.fixture
def few_iterations(tmp_path, monkeypatch):
    import byosm.series.models as models
    import byosm.series.rows as rows

    monkeypatch.setenv("BYOSM_CACHE", str(tmp_path))
    monkeypatch.setattr(models, "ITERATIONS", 30)
    monkeypatch.setattr(rows, "MAX_ROWS", 10_000)


def test_quantile_boosting_fits_a_model_per_level_and_reads_the_calendar_of_dated_series(few_iterations, monkeypatch):
    starts = [MONDAY_10PM + dt.timedelta(hours=13 * i) for i in range(4)]
    histories, horizon = [_weekdays(s, seed=i) for i, s in enumerate(starts)], 48
    train = [h[: -2 * horizon] for h in histories]
    clocks = [clock(s, "H") for s in starts]
    dated = QuantileBoosted()
    dated.fit([Group(train, 24, horizon, clocks)])
    assert [(m.loss, m.quantile) for m in dated.models] == [("quantile", q) for q in QUANTILES]
    out = dated.predict_many([h[:-horizon] for h in histories], 24, horizon, clocks)
    assert all(o.shape == (len(QUANTILES), horizon) and np.all(np.diff(o, axis=0) >= 0) for o in out)
    undated = QuantileBoosted()
    undated.fit([Group(train, 24, horizon)])
    plain = undated.predict_many([h[:-horizon] for h in histories], 24, horizon)
    median = QUANTILES.index(0.5)

    def error(forecasts):
        return np.mean([np.abs(f[median] - h[-horizon:]).mean() for f, h in zip(forecasts, histories, strict=True)])

    assert error(out) < 0.8 * error(plain)
    monkeypatch.setattr(QuantileBoosted, "_fit", lambda *a: pytest.fail("refit"))
    again = QuantileBoosted()
    again.fit([Group(train, 24, horizon, clocks)])
    assert len(again.models) == len(QUANTILES)


def test_a_live_example_sends_the_date_of_its_first_value_and_the_live_forecast_reads_it(ws, monkeypatch):
    import byosm.series.saved as saved

    monkeypatch.setattr(saved, "DOMAINS", {"demo": ["a"]})
    monkeypatch.setattr(saved, "_series", lambda config: [Series("s", "H", np.arange(24 * 100.0), MONDAY_10PM)])
    window, model = saved.example("demo", "a", 0), Recorder()
    first = MONDAY_10PM + dt.timedelta(hours=window["history"][0])
    assert np.datetime64(window["start"]) == np.datetime64(first)
    monkeypatch.setattr(saved, "load_model", lambda *a: model)
    saved.forecast(ws, "demo", "v1", window["history"], "H", None, window["start"])
    assert model.seen == [(24, [clock(first, "H")])]
