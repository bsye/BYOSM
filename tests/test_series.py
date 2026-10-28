import numpy as np
import pytest

from byosm.series.board import reports
from byosm.series.gifteval import Series, horizon, season, splits, unit, windows
from byosm.series.metrics import QUANTILES, crps, geomean, mase
from byosm.series.models import Boosted, Ensemble, Group, SeasonalNaive
from byosm.series.prediction import PredictCfg
from byosm.series.specialist import NAIVE

TRAINED = {"gradient boosting": SeasonalNaive()}


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
        model.fit([Group(histories, 24, 48)])
        forecasts = model.predict_many([t["history"] for t in tests], 24, 48)
        middle = QUANTILES.index(0.5)
        errors[model.name] = np.mean(
            [mase(t["history"], t["future"], f[middle], 24) for t, f in zip(tests, forecasts, strict=True)]
        )
    assert errors["gradient boosting"] < errors["seasonal naive"]


@pytest.mark.slow
def test_a_series_that_is_zero_half_the_time_is_not_forecast_as_all_zeros():
    t = np.arange(24 * 120)
    sun = [np.clip(np.sin(2 * np.pi * (t - 6) / 24), 0, None) * (100 + i) for i in range(4)]
    series = [Series(str(i), "H", v) for i, v in enumerate(sun)]
    cut = windows(series) * 48
    model = Boosted()
    model.fit([Group([s.values[: len(s.values) - cut] for s in series], 24, 48)])
    median = model.predict_many([series[0].values[: len(t) - cut]], 24, 48)[0][QUANTILES.index(0.5)]
    assert median.max() > 50


def test_the_patch_model_returns_sorted_quantiles_for_every_window(monkeypatch):
    import byosm.series.patch as patch

    monkeypatch.setattr(patch, "STEPS", 5)
    monkeypatch.setattr(patch, "BATCH", 8)
    histories = [_wave(seed=i) for i in range(3)]
    model = patch.PatchModel()
    monkeypatch.setattr(patch, "_device", lambda: "cpu")
    model.fit([Group(histories, 24, 48)])
    out = model.predict_many([h[:-48] for h in histories] + [np.full(10, np.nan)], 24, 48)
    assert len(out) == 4 and all(o.shape == (len(QUANTILES), 48) for o in out)
    assert all(np.all(np.diff(o, axis=0) >= 0) for o in out)


def test_a_pretrained_patch_model_starts_from_the_saved_weights(tmp_path, monkeypatch):
    import torch

    import byosm.series.patch as patch
    from byosm.series.patch_net import PatchNet

    saved = PatchNet(patch.BASE)
    torch.save(saved.state_dict(), tmp_path / "base.pt")
    monkeypatch.setattr(patch, "weights_path", lambda name="base": tmp_path / f"{name}.pt")
    monkeypatch.setattr(patch, "FINETUNE_STEPS", 1)
    monkeypatch.setattr(patch, "FINETUNE_LR", 1e-12)
    monkeypatch.setattr(patch, "BATCH", 4)
    monkeypatch.setattr(patch, "_device", lambda: "cpu")
    model = patch.PretrainedPatch()
    model.fit([Group([_wave()], 24, 30)])
    assert torch.allclose(model.net.embed.weight, saved.embed.weight, atol=1e-6)
    assert model.predict_many([_wave()[:-30]], 24, 30)[0].shape == (len(QUANTILES), 30)


@pytest.mark.slow
def test_a_smooth_series_gets_narrower_intervals_than_a_noisy_one_of_the_same_level():
    rng = np.random.default_rng(0)
    t = np.arange(600)
    smooth = 100 + 10 * np.sin(2 * np.pi * t / 24) + rng.normal(0, 0.5, len(t))
    noisy = 100 + 10 * np.sin(2 * np.pi * t / 24) + rng.normal(0, 20, len(t))
    model = Boosted()
    model.fit([Group([smooth, noisy] * 5, 24, 12)])
    calm, wild = model.predict_many([smooth, noisy], 24, 12)
    assert (calm[-1] - calm[0]).mean() * 5 < (wild[-1] - wild[0]).mean()


def test_a_domain_report_is_saved_after_each_recipe_so_results_show_as_they_arrive(ws, monkeypatch):
    import byosm.series.run as run

    seen = []
    score = {"mase": 1.0, "crps": 1.0, "train_s": 0.0, "ms_per_forecast": 0.1, "windows": [1.0]}

    def fit(configs, effort, log, progress, cfg):
        per = {c: {NAIVE: score} for c in configs}
        for pending in (["gradient boosting"], []):
            progress(per, pending)
            seen.append(next(r["pending"] for r in reports(ws) if r["domain"] == "demo"))
        return per, {}

    monkeypatch.setattr(run, "DOMAINS", {"demo": ["a", "b"]})
    monkeypatch.setattr(run, "fit_domain", fit)
    run.run_domain(ws, "demo", log=lambda _: None)
    assert seen == [["gradient boosting"], []]


def test_published_rival_scores_only_appear_where_our_seasonal_naive_matches_theirs(monkeypatch):
    import byosm.series.published as published

    def row(mase):
        return {published.MASE_COLUMN: str(mase), published.CRPS_COLUMN: "0.1"}

    tables = {
        published.NAIVE_MODEL: {"electricity/H/short": row(1.358), "restaurant/D/short": row(1.006)},
        **{m: {"electricity/H/short": row(0.8), "restaurant/D/short": row(0.7)} for m in published.RIVALS},
    }
    monkeypatch.setattr(published, "_table", tables.get)
    assert set(published.scores("electricity/H", 1.36)) == set(published.RIVALS)
    assert published.scores("restaurant", 1.006)[published.RIVALS[0]]["mase"] == 0.7
    assert published.scores("restaurant", 1.2) == {}


def _versioned_run(ws, monkeypatch, calib, test):
    import byosm.series.run as run

    naive = {"mase": 1.0, "crps": 1.0, "calib_mase": 1.0, "calib_crps": 1.0, "windows": [1.0, 1.0]}
    ours = {"mase": test, "crps": test, "calib_mase": calib, "calib_crps": calib, "windows": [test, test]}
    monkeypatch.setattr(run, "DOMAINS", {"demo": ["a"]})
    monkeypatch.setattr(
        run,
        "fit_domain",
        lambda configs, effort, log, progress, cfg: ({"a": {NAIVE: naive, "gradient boosting": ours}}, TRAINED),
    )
    monkeypatch.setattr("byosm.series.published.scores", lambda config, naive_mase: {})
    return run.run_domain(ws, "demo", log=lambda _: None)


def test_each_run_adds_a_version_and_promotes_it_only_if_it_is_better_on_calib(ws, monkeypatch):
    from byosm.series.specialist import summary

    first = _versioned_run(ws, monkeypatch, calib=0.8, test=0.9)
    assert first["promoted"] == "v1" and first["versions"][0]["recipe"] == "gradient boosting"
    second = _versioned_run(ws, monkeypatch, calib=0.85, test=0.7)
    assert [v["name"] for v in second["versions"]] == ["v1", "v2"] and second["promoted"] == "v1"
    card = summary(second)
    assert card["kind"] == "forecast" and card["score"] == 0.9 and card["higher_is_better"] is False


def test_each_version_saves_its_model_so_it_can_forecast_a_new_series_live(ws, monkeypatch):
    from byosm.series.saved import forecast, saved

    _versioned_run(ws, monkeypatch, calib=0.8, test=0.9)
    assert saved(ws, "demo", "v1") and not saved(ws, "demo", "v2")
    out = forecast(ws, "demo", "v1", [1.0, None, 3.0] * 30, "H")
    assert out["horizon"] == 48 and len(out["forecast"]) == len(QUANTILES) and len(out["forecast"][0]) == 48


def test_the_forecaster_routes_serve_a_test_window_and_a_live_forecast(ws, monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    import byosm.series.saved as saved
    from byosm.app.forecast_routes import forecast_router

    _versioned_run(ws, monkeypatch, calib=0.8, test=0.9)
    monkeypatch.setattr(saved, "DOMAINS", {"demo": ["a"]})
    monkeypatch.setattr(saved, "_series", lambda config: [Series("s", "H", _wave())])
    app = FastAPI()
    app.include_router(forecast_router(ws))
    client = TestClient(app)
    assert client.get("/api/series/demo").json()["live"] == "v1"
    window = client.get("/api/series/demo/example", params={"config": "a"}).json()
    assert len(window["future"]) == 48 and len(window["history"]) == saved.HISTORY
    body = client.post("/api/series/demo/forecast", json={"history": window["history"], "freq": window["freq"]}).json()
    assert body["version"] == "v1" and len(body["forecast"][0]) == 48
    assert client.get("/api/series/demo/example", params={"config": "b"}).status_code == 404


def test_one_model_per_recipe_is_trained_for_all_datasets_and_the_ensemble_averages_them(monkeypatch):
    import byosm.series.fit as fit

    rng = np.random.default_rng(0)
    t = np.arange(400.0)
    series = [Series(f"s{i}", "H", 10 + np.sin(2 * np.pi * t / 24) + rng.normal(0, 0.3, len(t))) for i in range(3)]

    fits = []

    class Flat:
        def __init__(self, name, level):
            self.name, self.level = name, level

        def fit(self, groups):
            fits.append((self.name, len(groups)))

        def predict_many(self, histories, season, horizon):
            return [np.full((len(QUANTILES), horizon), h[-1] + self.level) for h in histories]

    monkeypatch.setattr("byosm.series.scoring.load", lambda config: series)
    monkeypatch.setattr(
        fit,
        "candidates",
        lambda effort: [SeasonalNaive(), Flat("gradient boosting", 1.0), Flat("patch transformer", -1.0)],
    )
    events = []
    monkeypatch.setattr(fit, "emit", lambda kind, **data: events.append((kind, data["recipe"], data["index"])))
    out, trained = fit.fit_domain(["a", "b"], 0, lambda _: None, lambda per, pending: None, PredictCfg())
    assert [e[1:] for e in events] == [("seasonal naive", 1), ("gradient boosting", 2), ("patch transformer", 3)]
    assert set(trained) == {"gradient boosting", "patch transformer", Ensemble.name}
    assert fits == [("gradient boosting", 2), ("patch transformer", 2)]
    for row in out.values():
        assert row[Ensemble.name]["mase"] < min(row["gradient boosting"]["mase"], row["patch transformer"]["mase"])


def test_keep_training_stops_once_a_version_beats_the_best_rival(ws, monkeypatch):
    import byosm.series.run as run

    efforts, naive = [], {"mase": 1.0, "crps": 1.0, "calib_mase": 1.0, "calib_crps": 1.0, "windows": [1.0]}

    def one(configs, effort, log, progress, cfg):
        efforts.append(effort)
        ours = 0.9 - 0.1 * effort
        return {
            "a": {NAIVE: naive, "gradient boosting": {**naive, "mase": ours, "calib_mase": ours, "windows": [ours]}}
        }, TRAINED

    monkeypatch.setattr(run, "DOMAINS", {"demo": ["a"]})
    monkeypatch.setattr(run, "fit_domain", one)
    rival = {"Timestral-5M": {"mase": 0.75, "crps": 0.5, "published": True}}
    monkeypatch.setattr("byosm.series.published.scores", lambda config, naive_mase: rival)
    report = run.until_rival(ws, "demo", rounds=5, log=lambda _: None)
    assert efforts == [0, 1, 2] and report["promoted"] == "v3"


def test_the_rival_is_compared_on_the_datasets_both_were_scored_on_and_the_hardest_one_wins():
    from byosm.series.specialist import best_rival

    def row(ours, timestral=None):
        out = {"seasonal naive": {"mase": 1.0}, "patch transformer": {"mase": ours}}
        return {**out, "Timestral-209M": {"mase": timestral}} if timestral else out

    configs = {"a": row(0.7, 0.6), "b": row(0.8)}
    report = {"promoted": "v1", "versions": [{"name": "v1", "recipe": "patch transformer", "configs": configs}]}
    rival = best_rival(report)
    assert rival["name"] == "Timestral-209M" and rival["datasets"] == 1 and rival["of"] == 2
    assert rival["ours"] == pytest.approx(0.7) and rival["score"] == pytest.approx(0.6)


def test_the_board_shows_a_rival_scored_on_some_datasets_next_to_ours_on_the_same_ones():
    from byosm.series.board import promoted_board

    def row(ours, timestral=None):
        out = {"seasonal naive": {"mase": 1.0}, "patch transformer": {"mase": ours}}
        return {**out, "Timestral-5M": {"mase": timestral}} if timestral else out

    configs = {"a": row(0.7, 0.6), "b": row(0.8)}
    report = {"promoted": "v1", "versions": [{"name": "v1", "recipe": "patch transformer", "configs": configs}]}
    overall = promoted_board(report)["overall"]
    assert overall["Timestral-5M"]["datasets"] == 1 and overall["Timestral-5M"]["mase"] == pytest.approx(0.6)
    assert overall["Timestral-5M"]["ours_on_shared"] == pytest.approx(0.7)
    assert overall["patch transformer"]["datasets"] == 2 and "ours_on_shared" not in overall["patch transformer"]


def test_the_pretraining_corpus_draws_windows_from_single_and_multi_variate_series(tmp_path):
    from datasets import Dataset

    from byosm.series.pretrain import Corpus

    Dataset.from_dict({"target": [list(np.arange(300.0))]}).save_to_disk(tmp_path / "one")
    Dataset.from_dict({"target": [[list(np.arange(300.0)), list(np.arange(300.0) * 2)]]}).save_to_disk(
        tmp_path / "many"
    )
    paths = [str(next((tmp_path / d).glob("*.arrow"))) for d in ("one", "many")]
    corpus, rng = Corpus(paths, 120, 32), np.random.default_rng(0)
    drawn = [corpus.draw(rng) for _ in range(20)]
    assert all(d is not None and len(d) == 120 for d in drawn)
    assert any(d[1] - d[0] == 2 for d in drawn) and any(d[1] - d[0] == 1 for d in drawn)


def test_keep_training_stops_when_doubling_the_training_no_longer_helps(ws, monkeypatch):
    import byosm.series.run as run

    efforts, naive = [], {"mase": 1.0, "crps": 1.0, "calib_mase": 1.0, "calib_crps": 1.0, "windows": [1.0]}

    def one(configs, effort, log, progress, cfg):
        efforts.append(effort)
        ours = 0.85 - 0.001 * effort
        return {
            "a": {NAIVE: naive, "gradient boosting": {**naive, "mase": ours, "calib_mase": ours, "windows": [ours]}}
        }, TRAINED

    monkeypatch.setattr(run, "DOMAINS", {"demo": ["a"]})
    monkeypatch.setattr(run, "fit_domain", one)
    rival = {"Timestral-5M": {"mase": 0.5, "crps": 0.5, "published": True}}
    monkeypatch.setattr("byosm.series.published.scores", lambda config, naive_mase: rival)
    run.until_rival(ws, "demo", rounds=5, log=lambda _: None)
    assert efforts == [0, 1]


def test_forecaster_health_checks_naive_everywhere_agreement_precision_and_the_rival():
    from byosm.series.health import health

    def row(ours, naive=1.0, timestral=None):
        out = {"seasonal naive": {"mase": naive}, "patch transformer": {"mase": ours}}
        return {**out, "Timestral-5M": {"mase": timestral}} if timestral else out

    version = {"name": "v1", "recipe": "patch transformer", "calib": 0.80, "test": 0.82, "ci": [0.80, 0.84]}
    report = {"promoted": "v1", "versions": [{**version, "configs": {"a": row(0.8, timestral=0.7), "b": row(1.1)}}]}
    h = health(report)
    status = {c["id"]: (c["status"], c["value"]) for c in h["checks"]}
    assert status["robust"] == ("warn", "1 of 2 datasets") and status["overfit"][0] == "ok"
    assert status["rival"][0] == "fail" and status["certainty"] == ("ok", "±0.020")
    assert not h["ready"] and "Beats the best rival" not in h["blockers"]


def test_a_nearly_flat_window_cannot_blow_up_the_training_targets():
    from byosm.series.patch_net import SMALL, TARGET_CAP, batch, window

    flat = np.concatenate([np.full(600, 100.0) + np.random.default_rng(0).normal(0, 1e-6, 600), np.full(60, 160.0)])
    _, _, mean, std = window(flat[:600], SMALL.context)
    assert std >= 0.99 and abs(mean - 100.0) < 1e-3
    _, _, targets, _ = batch([(flat, 60)] * 4, SMALL, np.random.default_rng(1), "cpu")
    assert float(targets.abs().max()) <= TARGET_CAP


def test_only_a_short_small_forecaster_training_falls_back_to_the_cpu_and_the_rest_wait_for_the_gpu(
    tmp_path, monkeypatch
):
    import contextlib

    import byosm.series.patch as patch
    import byosm.train.gpu as gpu

    asked = []

    @contextlib.contextmanager
    def slot(device, log, fallback=None, key=None):
        asked.append(fallback)
        yield "cpu"

    monkeypatch.setattr(gpu, "gpu_slot", slot)
    monkeypatch.setattr(patch, "weights_path", lambda name="base": tmp_path / f"{name}.pt")
    monkeypatch.setattr(patch, "train", lambda *a, **k: None)
    monkeypatch.setattr(patch.torch, "load", lambda *a, **k: patch.PatchNet(patch.BASE).state_dict())
    for model in (patch.PatchModel(), patch.PretrainedPatch(), patch.PatchModel(effort=3)):
        model.fit([Group([_wave()], 24, 30)])
    assert asked == ["cpu", None, None]


def test_the_pretrained_recipe_uses_the_largest_base_that_has_been_pretrained(tmp_path, monkeypatch):
    import byosm.series.patch as patch

    monkeypatch.setattr(patch, "weights_path", lambda name="base": tmp_path / f"{name}.pt")
    assert patch.pretrained() is None
    (tmp_path / "base.pt").write_bytes(b"")
    assert patch.pretrained()[0] == "base"
    (tmp_path / "large.pt").write_bytes(b"")
    assert patch.pretrained() == ("large", patch.SIZES["large"]) and patch.PretrainedPatch().weights == "large"


def test_the_next_lever_names_the_smallest_base_not_pretrained_yet(tmp_path, monkeypatch):
    import byosm.series.patch as patch
    from byosm.series.run import _lever

    monkeypatch.setattr(patch, "weights_path", lambda name="base": tmp_path / f"{name}.pt")
    assert patch.missing_size() == "base" and "--size base" in _lever()
    (tmp_path / "base.pt").write_bytes(b"")
    assert "--size large" in _lever()
    (tmp_path / "large.pt").write_bytes(b"")
    assert patch.missing_size() is None and "series corpus" in _lever()


def _plateaued(ws, monkeypatch, upstream):
    import byosm.series.run as run

    efforts, naive = [], {"mase": 1.0, "crps": 1.0, "calib_mase": 1.0, "calib_crps": 1.0, "windows": [1.0]}

    def one(configs, effort, log, progress, cfg):
        efforts.append(effort)
        ours = 0.85 - 0.001 * effort
        return {
            "a": {NAIVE: naive, "gradient boosting": {**naive, "mase": ours, "calib_mase": ours, "windows": [ours]}}
        }, TRAINED

    monkeypatch.setattr(run, "DOMAINS", {"demo": ["a"]})
    monkeypatch.setattr(run, "fit_domain", one)
    monkeypatch.setattr(run, "_upstream", lambda: upstream)
    monkeypatch.setattr(run, "_lever", lambda: "the large base")
    rival = {"Timestral-5M": {"mase": 0.5, "crps": 0.5, "published": True}}
    monkeypatch.setattr("byosm.series.published.scores", lambda config, naive_mase: rival)
    report = run.until_rival(ws, "demo", rounds=5, log=lambda _: None)
    assert report[run.PLATEAU]["effort"] == 1
    return run, efforts


def test_keep_training_after_a_plateau_pretrains_the_missing_base_before_training_again(ws, monkeypatch):
    import byosm.series.patch as patch
    import byosm.series.pretrain as pretrain

    run, efforts = _plateaued(ws, monkeypatch, {"base": "base", "corpus": 3})
    pretrained = []
    monkeypatch.setattr(patch, "missing_size", lambda: "large")
    monkeypatch.setattr(pretrain, "pretrain", lambda log, steps, size: pretrained.append(size))
    run.until_rival(ws, "demo", rounds=1, log=lambda _: None)
    assert pretrained == ["large"] and efforts == [0, 1, 2]


def test_keep_training_after_a_plateau_with_every_base_pretrained_trains_nothing(ws, monkeypatch):
    import byosm.series.patch as patch

    run, efforts = _plateaued(ws, monkeypatch, {"base": "large", "corpus": 3})
    monkeypatch.setattr(patch, "missing_size", lambda: None)
    run.until_rival(ws, "demo", rounds=3, log=lambda _: None)
    assert efforts == [0, 1]


def test_the_rival_is_not_compared_with_models_the_promoted_version_was_not_scored_with():
    from byosm.series.specialist import best_rival

    current = {"a": {"seasonal naive": {"mase": 1.0}, "ensemble": {"mase": 0.5}, "Timestral-5M": {"mase": 0.6}}}
    report = {"promoted": "v1", "configs": current, "versions": [{"name": "v1", "recipe": "ensemble"}]}
    assert best_rival(report) is None


@pytest.mark.parametrize(("used", "fits"), [(50, 1), (300, 2)])
def test_boosting_that_stopped_early_is_reused_when_more_iterations_are_allowed(tmp_path, monkeypatch, used, fits):
    import byosm.series.models as models

    monkeypatch.setenv("BYOSM_CACHE", str(tmp_path))
    calls = []

    def fake(self, x, y, late, iterations, longest):
        calls.append(iterations)
        self.model, self.spread, self.used = "model", np.zeros((len(QUANTILES), longest)), [used, used]

    monkeypatch.setattr(models.Boosted, "_fit", fake)
    groups = [Group([_wave()], 24, 48)]
    models.Boosted(effort=0).fit(groups)
    again = models.Boosted(effort=1)
    again.fit(groups)
    assert len(calls) == fits and again.model == "model"
    models.Boosted(effort=1).fit(groups)
    assert len(calls) == fits


def test_calib_and_test_must_agree_on_every_dataset_not_only_on_average():
    from byosm.series.health import health

    def row(calib, test):
        naive = {"mase": 1.0, "calib_mase": 1.0}
        return {"seasonal naive": naive, "patch transformer": {"mase": test, "calib_mase": calib}}

    configs = {"electricity": row(0.70, 0.66), "solar": row(0.82, 1.10)}
    version = {"name": "v1", "recipe": "patch transformer", "calib": 0.76, "test": 0.80, "ci": [0.7, 0.9]}
    h = health({"promoted": "v1", "versions": [{**version, "configs": configs}]})
    agree = next(c for c in h["checks"] if c["id"] == "overfit")
    assert agree["status"] == "fail" and agree["value"] == "34% apart on solar"


def test_a_live_forecast_uses_the_newest_saved_version_when_the_promoted_one_has_no_model(ws):
    from byosm.series.saved import live_version, save

    report = {"promoted": "v1", "versions": [{"name": n} for n in ("v1", "v2", "v3")]}
    assert live_version(ws, "demo", report) is None
    save(ws, "demo", "v2", TRAINED["gradient boosting"])
    assert live_version(ws, "demo", report) == "v2"
    save(ws, "demo", "v1", TRAINED["gradient boosting"])
    assert live_version(ws, "demo", report) == "v1"
