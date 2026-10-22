from conftest import save_board

from byosm.health import health
from byosm.project import Project
from byosm.spec import EvalSuite, Label, TaskSpec


def _specialist(ws, rows):
    p = Project("qa").ensure()
    spec = TaskSpec(
        name="qa",
        description="is it a question",
        modality="text",
        labels=[Label(name="statement", definition="no"), Label(name="question", definition="yes")],
        baselines=["jev"],
        evals=[EvalSuite(name="builtin", primary=True), EvalSuite(name="llm-written", kind="generated")],
    )
    p.save_spec(spec)
    p.save_experiment("v1", meta={})
    p.promote("v1")
    save_board(p, rows)
    return p, spec


def _result(objective, kind="experiment", **extra):
    return {"kind": kind, "objective": objective, "n_test": 500, "n_calib": 200, **extra}


def test_a_model_with_no_rival_and_one_suite_is_not_ready(ws):
    p, spec = _specialist(ws, {"v1": {"builtin": _result(0.97, latency_ms=2.0)}})
    h = health(p, spec)
    status = {c["id"]: c["status"] for c in h["checks"]}
    assert status["rival"] == status["robust"] == status["overfit"] == "unknown"
    assert status["target"] == status["speed"] == "ok"
    assert not h["ready"] and "Beats the best rival" in h["blockers"]


def test_a_big_drop_on_another_suite_fails_robustness(ws):
    rows = {
        "v1": {"builtin": _result(0.97), "llm-written": _result(0.6)},
        "jev": {"builtin": _result(0.8, kind="baseline")},
    }
    p, spec = _specialist(ws, rows)
    checks = {c["id"]: c for c in health(p, spec)["checks"]}
    assert checks["robust"]["status"] == "fail" and checks["robust"]["value"] == "-0.370 on llm-written"
    assert checks["rival"]["status"] == "warn"


def test_no_promoted_score_gives_no_health(ws):
    p, spec = _specialist(ws, {})
    assert health(p, spec) is None
