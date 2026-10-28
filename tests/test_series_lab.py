import pickle

import numpy as np
import pytest
from conftest import Flat, save_forecaster

from byosm.lab.drafts import APPROVED, decide, draft, save_draft
from byosm.lab.owners import Forecaster, owner
from byosm.lab.plugin import Method
from byosm.series.gifteval import Series
from byosm.series.metrics import QUANTILES
from byosm.series.models import Group, SeasonalNaive
from byosm.series.patch_net import MAX_HORIZON, PatchNet, Size, picker, pinball, train
from byosm.series.prediction import FitCfg
from byosm.series.specialist import FIT

TINY = Size(context=32, patch=8, width=16, layers=1, heads=2)
BOOSTED, PATCH = "gradient boosting", "patch transformer"
JITTER = """import numpy as np


def window(values, context, rng):
    out = np.array(values, dtype=float)
    out[:context] = out[:context] * rng.uniform(0.9, 1.1)
    return out
"""


def _wave(n=400, seed=0):
    t = np.arange(n)
    return 10 + np.sin(2 * np.pi * t / 24) + np.random.default_rng(seed).normal(0, 0.3, n)


def _approved(f, code=JITTER):
    d = save_draft(f, "jitter", "scale noise", code, "test")
    decide(f, d["id"], APPROVED)
    return d["id"]


@pytest.fixture(autouse=True)
def isolated(ws, monkeypatch):
    monkeypatch.setenv("BYOSM_CACHE", str(ws / "cache"))
    monkeypatch.setattr("byosm.series.published.scores", lambda config, naive_mase: {})


def test_the_patch_transformer_trains_through_every_hook_it_is_given():
    seen = {"window": [], "loss": [], "after_step": []}

    def window(values, context, rng):
        seen["window"].append((len(values), context))
        return values

    def loss(pred, target, known, levels, step):
        seen["loss"].append((tuple(pred.shape), tuple(target.shape), levels.tolist(), step))
        return pinball(pred, target, known)

    hooks = {"window": window, "loss": loss, "after_step": lambda model, step, total: seen["after_step"].append(step)}
    train(PatchNet(TINY), picker([(_wave(), 12)], TINY.patch), 3, 1e-3, 4, "cpu", 0, hooks=hooks)
    assert seen["after_step"] == [0, 1, 2] and [s[-1] for s in seen["loss"]] == [0, 1, 2]
    shape, target, levels, _ = seen["loss"][0]
    assert shape == (4, len(QUANTILES), MAX_HORIZON) and target == (4, MAX_HORIZON)
    assert levels == pytest.approx(QUANTILES) and seen["window"][0] == (TINY.context + MAX_HORIZON, TINY.context)
    with pytest.raises(ValueError):
        train(
            PatchNet(TINY),
            picker([(_wave(), 12)], TINY.patch),
            1,
            1e-3,
            4,
            "cpu",
            0,
            hooks={"window": lambda v, c, r: v[1:]},
        )


def test_a_patch_model_with_a_lab_method_is_named_by_it_and_trains_with_its_approved_hooks(ws, monkeypatch):
    import byosm.series.patch as patch
    from byosm.series.fit import candidates, refreshed

    save_forecaster(ws, "demo")
    f = owner("demo")
    did = _approved(f)
    given = []
    monkeypatch.setattr(patch, "train", lambda *a: given.append(a[-1]))
    monkeypatch.setattr(patch, "_device", lambda: "cpu")
    model = patch.PatchModel(0, Method(f, did))
    model.fit([Group([_wave()], 24, 30)])
    assert model.name == f"{PATCH}, lab {did}" and set(given[0]) == {"window"}
    assert pickle.loads(pickle.dumps(model)).name == model.name
    names = [m.name for m in candidates(0, FitCfg(lab_method=did), f)]
    assert BOOSTED in names and f"{PATCH}, lab {did}" in names and PATCH not in names
    kept, fresh = refreshed([Flat(BOOSTED), Flat(PATCH)], 0, FitCfg(), FitCfg(lab_method=did), f)
    assert [m.name for m in kept] == [BOOSTED] and [m.name for m in fresh] == [model.name]
    with pytest.raises(PermissionError):
        patch.PatchModel(0, Method(Forecaster(f.ws, "other"), did)).fit([Group([_wave()], 24, 30)])


class Fitted(Flat):
    def __init__(self, name, level, trained, seasonal=False, fails=False):
        super().__init__(name, level)
        self.trained, self.seasonal, self.fails = trained, seasonal, fails

    def fit(self, groups):
        if self.fails:
            raise RuntimeError("loss is nan")
        self.trained.append(self.name)

    def predict_many(self, histories, season, horizon, clocks=None):
        if not self.seasonal:
            return super().predict_many(histories, season, horizon, clocks)
        return [np.tile(np.resize(h[-season:], horizon), (len(QUANTILES), 1)) for h in histories]


@pytest.fixture
def domain(ws, monkeypatch):
    import byosm.series.fit as fit
    import byosm.series.run as run
    from byosm.series.patch import lab_named

    trained, lab = [], {}

    def models(effort, options, owner_=None):
        method = Method(owner_, options.lab_method) if options.lab_method else None
        patch = Fitted(lab_named(PATCH, method), -0.5, trained, **(lab if method else {}))
        return [SeasonalNaive(), Fitted(BOOSTED, 0.5, trained), patch]

    monkeypatch.setattr(
        "byosm.series.scoring.load", lambda config: [Series(f"s{i}", "H", _wave(seed=i)) for i in range(3)]
    )
    monkeypatch.setattr(fit, "candidates", models)
    monkeypatch.setattr(fit, "emit", lambda kind, **data: None)
    monkeypatch.setattr(run, "DOMAINS", {"demo": ["a"]})
    run.until_rival(ws, "demo", rounds=1, log=lambda _: None)
    trained.clear()
    return trained, lab


@pytest.fixture
def revisions(monkeypatch):
    asked = []
    monkeypatch.setattr(
        "byosm.lab.author.write", lambda name, ask, log, paper, parent: asked.append((name, ask, parent))
    )
    return asked


@pytest.mark.parametrize("better", [True, False])
def test_a_lab_run_refits_only_the_patch_transformer_and_promotes_only_a_better_calib_score(
    ws, domain, revisions, better
):
    from byosm.lab.run import run
    from byosm.series.board import reports

    trained, lab = domain
    lab |= {"seasonal": True} if better else {}
    f = owner("demo")
    did = _approved(f)
    run("demo", did, lambda _: None)
    report = next(r for r in reports(ws) if r["domain"] == "demo")
    newest = report["versions"][-1]
    assert trained == [f"{PATCH}, lab {did}"] and newest[FIT]["lab_method"] == did and newest["name"] == "v2"
    assert draft(f, did)["result"] == {"experiment": "v2", "won": better, "calib": newest["calib"]}
    assert (report["promoted"] == "v2") is better
    assert [(n, parent) for n, _, parent in revisions] == ([] if better else [("demo", did)])


def test_a_method_the_promoted_version_already_trains_with_is_refused_without_an_error_or_a_revision(
    ws, domain, revisions
):
    from byosm.lab.drafts import RefusedError
    from byosm.lab.run import run

    _, lab = domain
    lab |= {"seasonal": True}
    f = owner("demo")
    did = _approved(f)
    run("demo", did, lambda _: None)
    with pytest.raises(RefusedError):
        run("demo", did, lambda _: None)
    assert draft(f, did)["error"] is None and revisions == []


def test_a_lab_run_that_fails_keeps_its_error_on_the_draft_and_asks_for_a_revision(ws, domain, revisions):
    from byosm.lab.run import run

    _, lab = domain
    lab |= {"fails": True}
    f = owner("demo")
    did = _approved(f)
    with pytest.raises(PermissionError):
        run("demo", save_draft(f, "other", "x", JITTER + "\n", "test")["id"], lambda _: None)
    with pytest.raises(RuntimeError):
        run("demo", did, lambda _: None)
    assert "loss is nan" in draft(f, did)["error"]
    assert len(revisions) == 1 and "loss is nan" in revisions[0][1] and revisions[0][2] == did
