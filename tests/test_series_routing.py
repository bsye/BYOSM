import numpy as np
from conftest import Flat

from byosm.series.gifteval import Series
from byosm.series.metrics import QUANTILES, summary
from byosm.series.models import SeasonalNaive
from byosm.series.prediction import Options, PredictCfg
from byosm.series.routing import Routed, route
from byosm.series.scoring import Prepared
from byosm.series.specialist import ROUTE, ROUTED

OURS = ["gradient boosting", "patch transformer"]


class KnowsSeason:
    def __init__(self, name, season):
        self.name, self.season = name, season

    def fit(self, groups):
        pass

    def predict_many(self, histories, season, horizon, clocks=None):
        miss = 0.5 if season == self.season else 50.0
        return [np.tile(h[-1] + np.arange(1, horizon + 1) + miss, (len(QUANTILES), 1)) for h in histories]


def _constant(value, steps=2):
    return np.full((len(QUANTILES), steps), value)


def test_routing_scores_calib_on_windows_other_than_the_ones_it_chose_on():
    calib = [{"history": np.array([0.0, 1.0, 0.0, 1.0]), "future": np.array([1.0, 1.0])} for _ in range(4)]
    p = Prepared("a", 1, 2, calib, calib, [])
    good, bad = _constant(1.0), _constant(3.0)
    made = {"a": ([good, bad, good, bad], [good] * 4), "b": ([bad, good, bad, good], [bad] * 4)}
    chosen, crossed, test = route(p, made)
    seasons = [1] * 4
    assert summary(calib, made[chosen][0], seasons)["mase"] == 1.0
    assert summary(calib, crossed, seasons)["mase"] == 2.0
    assert test is made[chosen][1]


def test_each_dataset_is_forecast_by_the_model_that_was_best_on_its_calib_windows(monkeypatch):
    import byosm.series.fit as fit
    from byosm.series.board import overall, promoted_board, stored
    from byosm.series.specialist import add_version, summary

    data = {"a": [Series(f"a{i}", "H", np.arange(400.0)) for i in range(2)]}
    data["b"] = [Series(f"b{i}", "D", np.arange(400.0)) for i in range(2)]
    monkeypatch.setattr("byosm.series.scoring.load", data.get)
    monkeypatch.setattr(fit, "emit", lambda kind, **data: None)
    models = [SeasonalNaive(), KnowsSeason(OURS[0], 24), KnowsSeason(OURS[1], 7)]
    monkeypatch.setattr(fit, "candidates", lambda effort, fit: models)
    per, trained = fit.fit_domain(
        ["a", "b"], 0, lambda _: None, lambda per, pending: None, Options(predict=PredictCfg(routing=True))
    )
    assert [per[c][ROUTED][ROUTE] for c in "ab"] == OURS
    assert all(per[c][ROUTED]["calib_mase"] == per[c][r]["calib_mase"] for c, r in zip("ab", OURS, strict=True))
    assert trained[ROUTED].for_config("b") is models[2]
    report = add_version({"domain": "demo", "configs": stored(per), "overall": overall(per)}, None)
    assert report["versions"][0]["recipe"] == ROUTED
    assert promoted_board(report)["routes"] == summary(report)["routes"] == {"a": OURS[0], "b": OURS[1]}


def test_without_routing_no_dataset_records_a_route(monkeypatch):
    import byosm.series.fit as fit

    monkeypatch.setattr("byosm.series.scoring.load", lambda config: [Series("s", "H", np.arange(400.0))])
    monkeypatch.setattr(fit, "emit", lambda kind, **data: None)
    monkeypatch.setattr(fit, "candidates", lambda effort, fit: [SeasonalNaive(), KnowsSeason(OURS[0], 24)])
    per, trained = fit.fit_domain(["a"], 0, lambda _: None, lambda per, pending: None, Options())
    assert ROUTED not in per["a"] and ROUTED not in trained


def test_a_live_forecast_uses_the_model_routed_to_the_dataset_it_names(ws):
    from byosm.series.saved import forecast, save

    routed = Routed({"up": Flat("up", 1.0), "down": Flat("down", -1.0)}, {"a": "up", "b": "down", "c": "up"})
    save(ws, "demo", "v1", routed)
    history = [5.0] * 100
    medians = {c: forecast(ws, "demo", "v1", history, "H", c)["forecast"][QUANTILES.index(0.5)][0] for c in "ab"}
    assert medians == {"a": 6.0, "b": 4.0}
    assert forecast(ws, "demo", "v1", history, "H")["forecast"][0][0] == 6.0
