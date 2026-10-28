import pytest
from conftest import save_board

from byosm import evals
from byosm.improve.bandit import catalogue_board
from byosm.improve.effects import EFFECT, effects, summary
from byosm.project import Project
from byosm.spec import Label, TaskSpec

EPOCHS = {"op": "set", "path": "train.epochs", "value": 12}
CALIB_ROWS = 40


def _specialist(name, modality="text"):
    p = Project(name).ensure()
    labels = [Label(name=n, definition="") for n in "ab"]
    p.save_spec(TaskSpec(name=name, description="", modality=modality, labels=labels, objective={"metric": "accuracy"}))
    return p


def _result(right, fp="f1", **extra):
    y = [i % 2 for i in range(CALIB_ROWS)]
    pred = [c if i < right else 1 - c for i, c in enumerate(y)]
    score = right / CALIB_ROWS
    units = {"calib": {"y": y, "pred": pred}}
    return {
        "kind": "experiment",
        "objective": score,
        "calib_objective": score,
        "fingerprint": fp,
        "units": units,
        **extra,
    }


def _built(p, exp, parent, right, fp="f1", ops=(EPOCHS,), **extra):
    proposal = {"parent": parent, "applied": [{"id": exp, "why": "w", "ops": list(ops)}]}
    p.save_experiment(exp, meta={"calib_held": True}, proposal=proposal if parent or ops else None)
    save_board(p, {exp: {"builtin": _result(right, fp, **extra)}})


def test_an_experiment_is_measured_against_its_parent_on_calib_with_a_paired_interval(ws):
    p = _specialist("a")
    _built(p, "v1", None, 30, ops=())
    _built(p, "v2", "v1", 36)
    e = effects(p, {"v2": "v1"})["v2"]
    assert e["parent"] == "v1" and e["delta"] == 0.15
    assert 0 < e["lo"] <= e["delta"] <= e["hi"]


def test_the_effect_is_computed_once_until_either_side_is_scored_again(ws, monkeypatch):
    p = _specialist("a")
    _built(p, "v1", None, 30, ops=())
    _built(p, "v2", "v1", 36)
    real, calls = evals.compare, []
    monkeypatch.setattr(evals, "compare", lambda *a: calls.append(a) or real(*a))
    effects(p, {"v2": "v1"})
    effects(p, {"v2": "v1"})
    assert len(calls) == 1 and p.report(EFFECT + "v2")["delta"] == 0.15
    save_board(p, {"v1": {"builtin": _result(32, identity="retrained")}})
    assert effects(p, {"v2": "v1"})["v2"]["delta"] == 0.1 and len(calls) == 2


def test_experiments_without_a_parent_on_the_same_suite_version_are_left_out_and_counted(ws):
    p = _specialist("a")
    _built(p, "v1", None, 30, ops=())
    _built(p, "v2", "v1", 36)
    _built(p, "v3", "v1", 38, fp="f2")
    _built(p, "v4", None, 38)
    _built(p, "v5", "gone", 38)
    effect = {r["id"]: r["calib_effect"] for r in catalogue_board(p.ws)}["schedule"]
    assert effect["measured"] == 1 and effect["left_out"] == 3 and effect["mean"] == 0.15
    assert effect["lo"] is None and effect["hi"] is None
    assert [e["experiment"] for e in effect["experiments"]] == ["v2"]


def test_a_family_mean_carries_a_student_t_interval_across_experiments():
    s = summary([0.1, 0.2, 0.3])
    assert s["measured"] == 3 and s["mean"] == 0.2
    assert s["lo"] == pytest.approx(-0.0484) and s["hi"] == pytest.approx(0.4484)
    assert summary([0.1]) == {"measured": 1, "mean": 0.1, "lo": None, "hi": None}
    assert summary([])["mean"] is None


def test_the_techniques_route_pools_effects_over_specialists_and_splits_them_by_kind(ws):
    from fastapi.testclient import TestClient

    from byosm.app import create_app

    for name, modality, right in (("a", "text", 36), ("c", "audio", 32)):
        p = _specialist(name, modality)
        _built(p, "v1", None, 30, ops=())
        _built(p, "v2", "v1", right)
    with TestClient(create_app(ws, "t"), base_url="http://127.0.0.1") as c:
        rows = c.get("/api/techniques", headers={"Authorization": "Bearer t"}).json()
    schedule = {r["id"]: r for r in rows}["schedule"]
    effect = schedule["calib_effect"]
    assert isinstance(schedule["effect"], str) and schedule["effect"]
    assert effect["measured"] == 2 and effect["mean"] == 0.1 and effect["lo"] < 0.1 < effect["hi"]
    assert [(k["modality"], k["measured"], k["mean"]) for k in effect["kinds"]] == [
        ("audio", 1, 0.05),
        ("text", 1, 0.15),
    ]
    assert {e["specialist"] for e in effect["experiments"]} == {"a", "c"}


def test_a_promotion_vote_measures_the_experiment_against_its_parent(ws):
    from byosm.promotion import maybe_promote

    p = _specialist("a")
    p.save_spec(p.spec().model_copy(update={"finalist_seeds": 1}))
    _built(p, "v1", None, 30, ops=())
    p.promote("v1")
    _built(p, "v2", "v1", 36)
    maybe_promote(p, "v2", lambda _: None)
    assert p.report(EFFECT + "v2")["delta"] == 0.15
