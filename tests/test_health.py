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


def test_a_missing_rival_counts_against_health_but_only_untested_inputs_block_production(ws):
    p, spec = _specialist(ws, {"v1": {"builtin": _result(0.97, latency_ms=2.0)}})
    h = health(p, spec)
    status = {c["id"]: c["status"] for c in h["checks"]}
    assert status["rival"] == status["robust"] == status["overfit"] == "unknown"
    assert status["target"] == status["speed"] == "ok"
    assert not h["ready"] and "Beats the best rival" not in h["blockers"] and "Holds on new inputs" in h["blockers"]


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


def test_the_lead_is_recomputed_when_a_new_rival_becomes_the_best(ws, monkeypatch):
    import byosm.health as health_module

    def fake_compare(spec, mine, theirs, split):
        delta = mine["objective"] - theirs["objective"]
        return {"delta": delta, "p_better": 0.5, "lo": delta, "hi": delta}

    monkeypatch.setattr(health_module, "compare", fake_compare)
    rows = {"v1": {"builtin": _result(0.9)}, "jev": {"builtin": _result(0.7, kind="baseline")}}
    p, spec = _specialist(ws, rows)
    assert {c["id"]: c for c in health(p, spec)["checks"]}["rival"]["value"].startswith("+0.200")
    p.add_baselines(["strong"])
    p.save_result("strong", "builtin", _result(0.88, kind="baseline"))
    assert {c["id"]: c for c in health(p, spec)["checks"]}["rival"]["value"].startswith("+0.020")


def test_a_specialist_bigger_than_its_rival_fails_the_size_check_without_blocking_production(ws):
    rows = {
        "v1": {"builtin": _result(0.97, params=110e6)},
        "jev": {"builtin": _result(0.9, kind="baseline", params=86e6)},
    }
    p, spec = _specialist(ws, rows)
    h = health(p, spec)
    size = next(c for c in h["checks"] if c["id"] == "size")
    assert size["status"] == "fail" and size["value"] == "1.3x bigger"
    assert "Smaller than the rival" not in h["blockers"]


def test_a_drop_on_an_llm_written_suite_is_judged_against_fine_tuned_rivals_on_the_same_suite(ws):
    rows = {
        "v1": {"builtin": _result(0.90), "llm-written": _result(0.54)},
        "hf:ref/emotion": {"builtin": _result(0.84, kind="baseline"), "llm-written": _result(0.53, kind="baseline")},
    }
    p, spec = _specialist(ws, rows)
    p.add_baselines(["hf:ref/emotion"])
    spec = p.spec()
    robust = next(c for c in health(p, spec)["checks"] if c["id"] == "robust")
    assert robust["status"] == "ok" and "vs 0.530 for rivals" in robust["value"]
