import numpy as np
import pytest
from conftest import Flat

from byosm.series.gifteval import Series
from byosm.series.models import Ensemble, SeasonalNaive
from byosm.series.multirate import MultiRate
from byosm.series.prediction import (
    MULTI_RATES,
    TURNED_ON,
    FitCfg,
    Options,
    PredictCfg,
    forecaster_techniques,
    next_technique,
)
from byosm.series.recipe import V1, V2
from byosm.series.routing import Routed
from byosm.series.specialist import FIT, NAIVE, PLATEAU, PREDICT, ROUTE, ROUTED

MULTI_RATE, ROUTING, QUANTILE, PRETRAINING = (
    "multi-rate-forecast",
    "forecast-routing",
    "quantile-boosting",
    "forecaster-pretraining",
)
BOOSTED, QUANTILE_BOOSTED, PATCH, PRETRAINED_V2 = (
    "gradient boosting",
    "gradient boosting, quantile",
    "patch transformer",
    "patch transformer, pretrained v2",
)


def _wave(n=400, seed=0):
    t = np.arange(n)
    return 10 + np.sin(2 * np.pi * t / 24) + np.random.default_rng(seed).normal(0, 0.3, n)


def _version(name, effort, fit=None, **predict):
    return {"name": name, "effort": effort, **Options(predict=PredictCfg(**predict), fit=fit or FitCfg()).model_dump()}


def test_the_forecaster_techniques_come_from_the_catalogue_and_each_turns_on_a_real_option():
    from byosm.improve.paths import allowed
    from byosm.improve.techniques import family
    from byosm.series.prediction import turned_on

    found = forecaster_techniques()
    assert [t.id for t in found] == [MULTI_RATE, ROUTING, QUANTILE, PRETRAINING]
    every = {f"{PREDICT}.{f}" for f in PredictCfg.model_fields} | {f"{FIT}.{f}" for f in FitCfg.model_fields}
    assert set(TURNED_ON) == every - {f"{FIT}.lab_method"}
    assert [turned_on(Options(), t) for t in found] == [
        Options(predict=PredictCfg(rates=MULTI_RATES)),
        Options(predict=PredictCfg(routing=True)),
        Options(fit=FitCfg(quantile_boosting=True)),
        Options(fit=FitCfg(pretraining=V2.name)),
    ]
    assert allowed("fit.pretraining") and family([{"path": "fit.quantile_boosting"}]) == QUANTILE
    with pytest.raises(ValueError):
        FitCfg(pretraining="v9")


def test_the_next_technique_is_one_that_no_version_at_the_promoted_effort_has_tried():
    report = {"promoted": "v1", "versions": [_version("v1", 0)]}
    assert next_technique(report) == (MULTI_RATE, Options(predict=PredictCfg(rates=MULTI_RATES)))
    report["versions"].append(_version("v2", 0, rates=MULTI_RATES))
    assert next_technique(report) == (ROUTING, Options(predict=PredictCfg(routing=True)))
    report["versions"].append(_version("v3", 0, routing=True))
    report["promoted"] = "v3"
    both = PredictCfg(rates=MULTI_RATES, routing=True)
    assert next_technique(report) == (MULTI_RATE, Options(predict=both))
    report["versions"].append(_version("v4", 0, rates=MULTI_RATES, routing=True))
    assert next_technique(report) == (
        QUANTILE,
        Options(predict=PredictCfg(routing=True), fit=FitCfg(quantile_boosting=True)),
    )
    report["versions"].append(_version("v5", 1, routing=True))
    report["promoted"] = "v5"
    assert next_technique(report) == (MULTI_RATE, Options(predict=both))


def test_pretraining_v2_is_tried_only_once_doubling_the_training_stopped_helping():
    tried = FitCfg(quantile_boosting=True)
    report = {"promoted": "v1", "versions": [_version("v1", 0, tried), _version("v2", 0, rates=MULTI_RATES)]}
    report["versions"] += [_version("v3", 0, routing=True), _version("v4", 0, tried, rates=MULTI_RATES)]
    report["versions"] += [_version("v5", 0, tried, routing=True)]
    assert next_technique(report) is None
    report[PLATEAU] = {"effort": 0}
    assert next_technique(report) == (PRETRAINING, Options(fit=FitCfg(pretraining=V2.name, quantile_boosting=True)))


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
            BOOSTED: {"train_s": 12.0},
            "Timestral-209M": {"mase": 0.5},
            "Timestral-5M": {"mase": 0.4, "published": True},
        }
    }
    per, trained = rescore(["a"], ([Flat(BOOSTED)], []), parent, lambda _: None, lambda *a: None, Options())
    assert per["a"][BOOSTED]["train_s"] == 12.0 and per["a"]["Timestral-209M"] == {"mase": 0.5}
    assert NAIVE in per["a"] and "Timestral-5M" not in per["a"] and list(trained) == [BOOSTED]


def test_an_option_that_changes_a_model_refits_only_that_model_and_keeps_the_others():
    from byosm.series.fit import refreshed

    saved = [Flat(BOOSTED), Flat(PATCH)]
    kept, fresh = refreshed(saved, 0, FitCfg(), FitCfg(quantile_boosting=True))
    assert kept == saved[1:] and [m.name for m in fresh] == [QUANTILE_BOOSTED]
    assert refreshed(saved, 0, FitCfg(), FitCfg()) == (saved, [])


class Fitted(Flat):
    def __init__(self, name, level, effort, trained):
        super().__init__(name, level)
        self.effort, self.trained = effort, trained

    def fit(self, groups):
        self.trained.append((self.name, self.effort))


@pytest.fixture
def domain(ws, monkeypatch):
    import byosm.series.fit as fit
    import byosm.series.run as run

    trained = []

    def models(effort, options, lab=None):
        boosted = QUANTILE_BOOSTED if options.quantile_boosting else BOOSTED
        ours = [Fitted(boosted, 0.5, effort, trained), Fitted(PATCH, -0.5, effort, trained)]
        base = [Fitted(PRETRAINED_V2, 3.0, effort, trained)] if options.recipe == V2 else []
        return [SeasonalNaive(), *ours, *base]

    series = [Series(f"s{i}", "H", _wave(seed=i)) for i in range(3)]
    monkeypatch.setenv("BYOSM_CACHE", str(ws / "cache"))
    monkeypatch.setattr("byosm.series.scoring.load", lambda config: series)
    monkeypatch.setattr(fit, "candidates", models)
    monkeypatch.setattr(fit, "emit", lambda kind, **data: None)
    monkeypatch.setattr(run, "DOMAINS", {"demo": ["a"]})
    monkeypatch.setattr("byosm.series.published.scores", lambda config, naive_mase: {})
    return run, trained


def test_keep_training_tries_each_technique_on_the_saved_models_before_training_again(ws, domain):
    run, trained = domain
    report = run.until_rival(ws, "demo", rounds=4, log=lambda _: None)
    versions = report["versions"]
    assert [v["effort"] for v in versions] == [0, 0, 0, 0]
    assert [v[PREDICT]["rates"] for v in versions[:2]] == [[1], MULTI_RATES] and versions[2][PREDICT]["routing"]
    assert ROUTE in versions[2]["configs"]["a"][ROUTED]
    assert trained == [(BOOSTED, 0), (PATCH, 0), (QUANTILE_BOOSTED, 0)] and versions[3][FIT]["quantile_boosting"]
    assert QUANTILE_BOOSTED in versions[3]["configs"]["a"] and BOOSTED not in versions[3]["configs"]["a"]


def test_a_forecaster_that_stopped_gaining_from_training_still_tries_its_untried_techniques(ws, domain, monkeypatch):
    import byosm.series.patch as patch

    run, trained = domain
    monkeypatch.setattr(run, "upstream", lambda recipe=None: {"base": "large", "corpus": 3})
    monkeypatch.setattr(patch, "missing_size", lambda recipe=None: None)
    first = run.until_rival(ws, "demo", rounds=1, log=lambda _: None)
    run._mark(ws, first, {"effort": 0, "upstream": run.upstream(), "lever": "more data"})
    report = run.until_rival(ws, "demo", rounds=2, log=lambda _: None)
    assert {e for _, e in trained} == {0} and len(report["versions"]) == 3 and report[run.PLATEAU]["effort"] == 0


def test_after_a_plateau_keep_training_tries_the_v2_recipe_once_before_pretraining_a_bigger_base(
    ws, domain, monkeypatch
):
    import byosm.series.patch as patch
    import byosm.series.pretrain as pretrain

    run, trained = domain
    pretrained = []
    monkeypatch.setattr(run, "upstream", lambda recipe=None: {"base": "base", "corpus": 0})
    monkeypatch.setattr(patch, "pretrained", lambda recipe=V1: None)
    monkeypatch.setattr(patch, "missing_size", lambda recipe=V1: "large" if recipe == V1 else "base")
    monkeypatch.setattr(pretrain, "pretrain", lambda log, steps, size, recipe: pretrained.append((size, recipe.name)))
    first = run.until_rival(ws, "demo", rounds=1, log=lambda _: None)
    run._mark(ws, first, {"effort": 0, "upstream": run.upstream(), "lever": "x"})
    report = run.until_rival(ws, "demo", rounds=8, log=lambda _: None)
    assert pretrained == [("base", "v2"), ("large", "v1")] and trained.count((PRETRAINED_V2, 0)) == 1
    tried = [v for v in report["versions"] if v[FIT]["pretraining"] == V2.name]
    assert [v["effort"] for v in tried] == [0] and PRETRAINED_V2 in tried[0]["configs"]["a"]
    assert (PATCH, 1) in trained and report[run.PLATEAU]["effort"] == 1
