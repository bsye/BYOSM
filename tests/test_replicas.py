import numpy as np
import pytest

from byosm.replicas import beyond_seed_noise, recipe_of, replica
from byosm.resample import bootstrap_runs
from byosm.spec import Label, TaskSpec

HELD = {"calib_held": True}


def _spec(**kw):
    labels = [Label(name="a", definition=""), Label(name="b", definition="")]
    return TaskSpec(name="tc", description="", modality="text", labels=labels, **kw)


def _result(calib):
    return {"kind": "experiment", "objective": calib, "calib_objective": calib, "fingerprint": "f"}


def test_a_replica_is_named_after_its_recipe_and_seed():
    assert replica("v3", 2) == "v3-s2" and recipe_of("v3-s2") == "v3"
    assert recipe_of("distill-v1") is None and recipe_of("v3") is None


def test_a_lead_counts_only_beyond_the_measured_seed_noise():
    assert beyond_seed_noise([0.95], [0.90], None)
    assert not beyond_seed_noise([0.95], [0.90], 0.04)
    assert beyond_seed_noise([0.95, 0.951, 0.949], [0.90, 0.901, 0.899], 0.01)


def test_the_seed_averaged_bootstrap_of_identical_runs_matches_the_single_run():
    rng = np.random.default_rng(0)
    a, b = rng.integers(0, 2, (200, 1)), rng.integers(0, 2, (200, 1))
    one = bootstrap_runs([a], [b], lambda s: float(s[0]))
    three = bootstrap_runs([a, a, a], [b], lambda s: float(s[0]))
    assert one == pytest.approx(three)
    with pytest.raises(ValueError):
        bootstrap_runs([a], [b[:10]], lambda s: float(s[0]))


def test_the_seed_sd_pools_every_recipe_with_more_than_one_clean_run(ws):
    from conftest import save_board

    from byosm.project import Project
    from byosm.replicas import seed_sd

    p = Project("tc").ensure()
    p.save_spec(_spec())
    values = {"v1": 0.80, "v1-s1": 0.82, "v2": 0.90, "v2-s1": 0.92, "v3": 0.5}
    for e in values:
        p.save_experiment(e, meta=HELD)
    p.save_experiment("v4", meta={})
    save_board(p, {e: {"builtin": _result(v)} for e, v in values.items()} | {"v4": {"builtin": _result(0.1)}})
    assert seed_sd(p, "builtin") == pytest.approx(np.sqrt(2e-4))


def test_a_single_lucky_seed_is_replicated_and_loses_on_the_seed_average(ws, monkeypatch):
    from conftest import save_board

    from byosm import evals, pipeline
    from byosm.project import Project
    from byosm.promotion import maybe_promote

    p = Project("tc").ensure()
    p.save_spec(_spec())
    for e in ("v1", "v1-s1", "v1-s2", "v2"):
        p.save_experiment(e, meta=HELD)
    p.promote("v1", "test")
    save_board(p, {"v1": {"builtin": _result(0.90)}, "v1-s1": {"builtin": _result(0.91)}})
    save_board(p, {"v1-s2": {"builtin": _result(0.89)}, "v2": {"builtin": _result(0.93)}})
    built = []

    def fake_build(name, exp, log, promote=True, suites=None):
        assert not promote and suites == ["builtin"]
        built.append(exp)
        p.save_experiment(exp, meta=HELD)
        save_board(p, {exp: {"builtin": _result(0.88)}})

    monkeypatch.setattr(pipeline, "build", fake_build)
    rest = []
    monkeypatch.setattr(evals, "run", lambda name, experiments, baselines, log: rest.append((experiments, baselines)))
    said = []
    maybe_promote(p, "v2", said.append)
    assert built == ["v2-s1", "v2-s2"]
    assert p.promoted == "v1"
    assert "over 3 vs 3 seeds" in said[-2] and said[-1].startswith("[seeds] scoring v2-s1, v2-s2")
    assert rest == [(["v2-s1", "v2-s2"], [])]


def test_the_seed_spread_groups_clean_runs_of_a_recipe_and_always_shows_the_promoted_one(ws):
    from conftest import save_board

    from byosm.project import Project
    from byosm.replicas import seed_spread

    p = Project("tc").ensure()
    p.save_spec(_spec())
    for e in ("v1", "v2", "v2-s1", "v2-soup", "v3", "v4", "v4-s1"):
        p.save_experiment(e, meta=HELD)
    p.save_experiment("v4-s2", meta={})
    p.promote("v1", "test")
    stale = {**_result(0.99), "fingerprint": "old"}
    save_board(
        p, {"v1": {"builtin": _result(0.90)}, "v2": {"builtin": _result(0.93)}, "v2-s1": {"builtin": _result(0.91)}}
    )
    save_board(
        p, {"v2-soup": {"builtin": _result(0.95)}, "v3": {"builtin": _result(0.80)}, "v4": {"builtin": _result(0.85)}}
    )
    save_board(p, {"v4-s1": {"builtin": stale}, "v4-s2": {"builtin": _result(0.86)}})
    found = seed_spread(p)
    assert found["suite"] == "builtin"
    assert [(r["recipe"], r["promoted"], [u["id"] for u in r["runs"]]) for r in found["recipes"]] == [
        ("v2", False, ["v2", "v2-s1"]),
        ("v1", True, ["v1"]),
    ]
    assert found["recipes"][0]["mean"] == pytest.approx(0.92)
