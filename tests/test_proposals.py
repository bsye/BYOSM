import pytest

from byosm import export, hub, improve
from byosm.auto import autopilot, loop
from byosm.auto.state import Log, Session
from byosm.improve import proposals
from byosm.project import Project
from byosm.spec import Label, TaskSpec

LOST = {"won": False, "exp": "v1", "ours": 0.5, "suite": "builtin", "best_baseline": ("hf:x", 0.9)}


def make_project(ws):
    p = Project("t").ensure()
    spec = TaskSpec(
        name="t",
        description="",
        modality="text",
        labels=[Label(name="a", definition=""), Label(name="b", definition="")],
    )
    p.save_spec(spec)
    p.exp_dir("v1").mkdir()
    p.save_experiment("v1", spec=spec)
    p.save_experiment("v1", meta={"params": 50e6})
    p.promote("v1")
    return p, spec


def session(p, spec):
    return Session(p.name, p, Log(p, lambda m: None), spec, 0.0, float("inf"), lambda m: None)


def test_duplicate_llm_ids_are_dropped_and_apply_works(ws, monkeypatch):
    p, spec = make_project(ws)
    rule = {"id": "llm-a", "why": "rule", "ops": [{"op": "set", "path": "train.epochs", "value": 3}]}
    first = {"id": "llm-a", "why": "first", "ops": [{"op": "set", "path": "train.epochs", "value": 4}]}
    second = {"id": "llm-a", "why": "second", "ops": [{"op": "set", "path": "train.epochs", "value": 5}]}
    other = {"id": "llm-b", "why": "other", "ops": [{"op": "set", "path": "train.epochs", "value": 6}]}
    monkeypatch.setattr(proposals, "diagnose", lambda name: {"spec": spec, "experiment": "v1", "train_seconds": 0})
    monkeypatch.setattr(proposals, "rule_proposals", lambda dg: [rule])
    monkeypatch.setattr(proposals, "llm_proposals", lambda dg, log: [first, second, other])
    _, props = improve.propose("t", log=lambda m: None)
    assert [pr["why"] for pr in props] == ["rule", "other"]
    exp = improve.apply("t", ["llm-a"])
    assert p.exp_spec(exp).train.epochs == 3


def test_proposal_whose_ops_do_not_apply_is_skipped(ws, monkeypatch):
    p, spec = make_project(ws)
    broken = {"id": "llm-x", "why": "bad path", "ops": [], "cost": {"minutes": float("nan"), "new_items": 0}}
    monkeypatch.setattr(loop, "standing", lambda name, margin: LOST)
    monkeypatch.setattr(improve, "propose", lambda name, use_llm, log: ({}, [broken]))

    def fail(*a, **k):
        raise AssertionError("broken proposal was applied")

    monkeypatch.setattr(improve, "apply", fail)
    s = session(p, spec)
    assert loop.improve_until_stop(s, 8, False) == "no untried proposals left"
    skips = [d for d in s.log.decisions if d["step"] == "skip"]
    assert skips[0]["choice"] == "llm-x" and "do not apply" in skips[0]["why"]


@pytest.mark.parametrize("won", [False, True])
def test_every_model_is_distilled_but_only_a_winner_is_published(ws, monkeypatch, won):
    p, spec = make_project(ws)
    calls = []
    monkeypatch.setattr(autopilot, "improve_until_stop", lambda s, n, use_llm: "plateau")
    monkeypatch.setattr(autopilot, "standing", lambda name, margin: {**LOST, "won": won})
    monkeypatch.setattr(autopilot, "_distill", lambda s, m: calls.append("distill"))
    monkeypatch.setattr(export, "export", lambda name, log: calls.append("export") or p.root / "export")
    monkeypatch.setattr(hub, "publish", lambda *a, **k: calls.append("publish"))
    monkeypatch.setattr(autopilot, "_report", lambda p, decisions, final: None)
    autopilot.run("t", publish="me/t", echo=lambda m: None)
    decisions = p.report("autopilot")["decisions"]
    assert calls == (["distill", "export", "publish"] if won else ["distill", "export"])
    skipped = {d["step"] for d in decisions if "did not win" in d["why"]}
    assert skipped == (set() if won else {"export", "publish"})


def test_won_is_rechecked_after_distill(ws, monkeypatch):
    p, spec = make_project(ws)
    calls = []
    wins = iter([False])
    monkeypatch.setattr(autopilot, "improve_until_stop", lambda s, n, use_llm: "won")
    monkeypatch.setattr(autopilot, "standing", lambda name, margin: {**LOST, "won": next(wins, False)})
    monkeypatch.setattr(autopilot, "_distill", lambda s, m: calls.append("distill"))
    monkeypatch.setattr(export, "export", lambda name, log: calls.append("export") or p.root / "export")
    monkeypatch.setattr(hub, "publish", lambda *a, **k: calls.append("publish"))
    monkeypatch.setattr(autopilot, "_report", lambda p, decisions, final: None)
    final = autopilot.run("t", publish="me/t", echo=lambda m: None)
    assert calls == ["distill", "export"] and not final["won"]


@pytest.mark.parametrize("student", [0.80, 0.90])
def test_a_student_is_kept_when_it_scores_within_the_tolerance_of_its_teacher(ws, monkeypatch, student):
    from byosm import distill, pipeline

    p, spec = make_project(ws)

    def create(name, teacher):
        p.exp_dir("v2").mkdir()
        p.save_experiment("v2", meta={"params": 5e6})
        return "v2"

    monkeypatch.setattr(distill, "create", create)
    monkeypatch.setattr(pipeline, "build", lambda name, exp, echo: p.promote(exp))
    monkeypatch.setattr(autopilot, "standing", lambda name, margin: {**LOST, "ours": student})
    s = session(p, spec)
    autopilot._distill_student(s, {**LOST, "ours": 0.905}, {"params": 50e6})
    assert p.promoted == ("v2" if student >= 0.905 - autopilot.DISTILL_TOLERANCE else "v1")


def test_an_experiment_that_never_finished_is_built_before_improving(ws, monkeypatch):
    p, spec = make_project(ws)
    p.save_experiment("v2", spec=spec)
    p.promote(None)
    calls = []
    monkeypatch.setattr(autopilot, "missing_labels", lambda spec: False)
    monkeypatch.setattr(autopilot, "_build_v1", lambda s: calls.append("build"))
    monkeypatch.setattr(autopilot, "improve_until_stop", lambda s, n, use_llm: calls.append("improve") or "plateau")
    monkeypatch.setattr(autopilot, "standing", lambda name, margin: LOST)
    monkeypatch.setattr(autopilot, "_finish", lambda *a: None)
    monkeypatch.setattr(autopilot, "_report", lambda p, decisions, final: None)
    autopilot.run("t", echo=lambda m: None)
    assert calls == ["build", "improve"]


def test_next_steps_that_run_the_same_command_are_shown_once_in_their_first_place():
    from byosm.next_steps import unique

    steps = [
        {"args": ["eval", "q"], "title": "a"},
        {"args": ["improve", "q"], "title": "b"},
        {"args": ["eval", "q"], "title": "c"},
    ]
    assert [s["title"] for s in unique(steps)] == ["a", "b"]
