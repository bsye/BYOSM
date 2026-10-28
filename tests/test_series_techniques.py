import numpy as np
import pytest
from conftest import Flat

from byosm.series.gifteval import Series
from byosm.series.models import Ensemble, SeasonalNaive
from byosm.series.multirate import MultiRate
from byosm.series.prediction import MULTI_RATES, TURNED_ON, PredictCfg, forecaster_techniques, next_technique
from byosm.series.routing import Routed
from byosm.series.specialist import NAIVE, PREDICT, ROUTE, ROUTED

MULTI_RATE, ROUTING = "multi-rate-forecast", "forecast-routing"


def _wave(n=400, seed=0):
    t = np.arange(n)
    return 10 + np.sin(2 * np.pi * t / 24) + np.random.default_rng(seed).normal(0, 0.3, n)


def _version(name, effort, **predict):
    return {"name": name, "effort": effort, PREDICT: PredictCfg(**predict).model_dump()}


def test_the_forecaster_techniques_come_from_the_catalogue_and_each_turns_on_a_real_prediction_option():
    from byosm.improve.paths import allowed
    from byosm.improve.techniques import family
    from byosm.series.prediction import turned_on

    found = forecaster_techniques()
    assert [t.id for t in found] == [MULTI_RATE, ROUTING] and set(TURNED_ON) == set(PredictCfg.model_fields)
    assert [turned_on(PredictCfg(), t) for t in found] == [PredictCfg(rates=MULTI_RATES), PredictCfg(routing=True)]
    assert allowed("predict.rates") and family([{"path": "predict.routing"}]) == ROUTING


def test_the_next_technique_is_one_that_no_version_at_the_promoted_effort_has_tried():
    report = {"promoted": "v1", "versions": [_version("v1", 0)]}
    assert next_technique(report) == (MULTI_RATE, PredictCfg(rates=MULTI_RATES))
    report["versions"].append(_version("v2", 0, rates=MULTI_RATES))
    assert next_technique(report) == (ROUTING, PredictCfg(routing=True))
    report["versions"].append(_version("v3", 0, routing=True))
    report["promoted"] = "v3"
    assert next_technique(report) == (MULTI_RATE, PredictCfg(rates=MULTI_RATES, routing=True))
    report["versions"].append(_version("v4", 0, rates=MULTI_RATES, routing=True))
    assert next_technique(report) is None
    report["versions"].append(_version("v5", 1, routing=True))
    report["promoted"] = "v5"
    assert next_technique(report) == (MULTI_RATE, PredictCfg(rates=MULTI_RATES, routing=True))


def test_the_trained_models_are_recovered_from_any_saved_version_without_the_baseline():
    from byosm.series.combine import members

    a, b = Flat("a"), Flat("b")
    saved = {Ensemble.name: Ensemble([a, b]), "twin": MultiRate(a, MULTI_RATES), NAIVE: SeasonalNaive()}
    routed = Routed(saved, {"x": NAIVE})
    assert members(routed) == [a, b] and members(SeasonalNaive()) == [] and members(b) == [b]


def test_rescoring_saved_models_keeps_their_training_time_and_the_rivals_scored_on_the_same_windows(monkeypatch):
    from byosm.series.fit import rescore

    monkeypatch.setattr("byosm.series.scoring.load", lambda config: [Series("s", "H", _wave())])
    parent = {
        "a": {
            "gradient boosting": {"train_s": 12.0},
            "Timestral-209M": {"mase": 0.5},
            "Timestral-5M": {"mase": 0.4, "published": True},
        }
    }
    per, trained = rescore(["a"], [Flat("gradient boosting")], parent, lambda _: None, lambda *a: None, PredictCfg())
    assert per["a"]["gradient boosting"]["train_s"] == 12.0 and per["a"]["Timestral-209M"] == {"mase": 0.5}
    assert NAIVE in per["a"] and "Timestral-5M" not in per["a"] and list(trained) == ["gradient boosting"]


@pytest.fixture
def domain(ws, monkeypatch):
    import byosm.series.fit as fit
    import byosm.series.run as run

    efforts = []

    def models(effort):
        efforts.append(effort)
        return [SeasonalNaive(), Flat("gradient boosting", 0.5), Flat("patch transformer", -0.5)]

    series = [Series(f"s{i}", "H", _wave(seed=i)) for i in range(3)]
    monkeypatch.setattr("byosm.series.scoring.load", lambda config: series)
    monkeypatch.setattr(fit, "candidates", models)
    monkeypatch.setattr(fit, "emit", lambda kind, **data: None)
    monkeypatch.setattr(run, "DOMAINS", {"demo": ["a"]})
    monkeypatch.setattr("byosm.series.published.scores", lambda config, naive_mase: {})
    return run, efforts


def test_keep_training_tries_each_technique_on_the_saved_models_before_training_again(ws, domain):
    run, efforts = domain
    report = run.until_rival(ws, "demo", rounds=3, log=lambda _: None)
    versions = report["versions"]
    assert efforts == [0] and [v["effort"] for v in versions] == [0, 0, 0]
    assert [v[PREDICT]["rates"] for v in versions[:2]] == [[1], MULTI_RATES] and versions[2][PREDICT]["routing"]
    assert ROUTE in versions[2]["configs"]["a"][ROUTED]


def test_a_forecaster_that_stopped_gaining_from_training_still_tries_its_untried_techniques(ws, domain, monkeypatch):
    import byosm.series.patch as patch

    run, efforts = domain
    monkeypatch.setattr(run, "_upstream", lambda: {"base": "large", "corpus": 3})
    monkeypatch.setattr(patch, "missing_size", lambda: None)
    first = run.until_rival(ws, "demo", rounds=1, log=lambda _: None)
    run._mark(ws, first, {"effort": 0, "upstream": run._upstream(), "lever": "more data"})
    report = run.until_rival(ws, "demo", rounds=2, log=lambda _: None)
    assert efforts == [0] and len(report["versions"]) == 3 and report[run.PLATEAU]["effort"] == 0
