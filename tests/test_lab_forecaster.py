import pickle

import pytest
from conftest import save_forecaster

from byosm.lab.contract import AFTER_STEP, CLASSIFIER, CONTRACTS, FORECASTER, RULES
from byosm.lab.drafts import APPROVED, decide, drafts, save_draft
from byosm.lab.owners import Forecaster, kind_of, owner
from byosm.lab.plugin import Method, load
from byosm.lab.static import problems

JITTER = """import numpy as np


def window(values, context, rng):
    out = np.array(values, dtype=float)
    out[:context] = out[:context] * rng.uniform(0.9, 1.1)
    return out


def after_step(model, step, total):
    return None
"""


@pytest.fixture(autouse=True)
def no_published_scores(monkeypatch):
    monkeypatch.setattr("byosm.series.published.scores", lambda config, naive_mase: {})


def test_a_forecaster_method_is_checked_against_the_forecaster_contract():
    assert problems(JITTER, FORECASTER) == []
    assert "define at least one of window, loss, after_step" in problems("def rows(t, r):\n    return t", FORECASTER)
    assert problems("def window(v, c, r):\n    return open('x')", FORECASTER)
    texts = {kind: c.text for kind, c in CONTRACTS.items()}
    assert "def window(" in texts[FORECASTER] and "logits" not in texts[FORECASTER]
    assert "def rows(" in texts[CLASSIFIER] and "def window(" not in texts[CLASSIFIER]
    assert all(RULES in t and AFTER_STEP in t and "numpy, random" in t for t in texts.values())


def test_a_name_resolves_to_a_specialist_or_a_forecaster_that_has_run(ws):
    from conftest import promoted_specialist

    promoted_specialist("qa")
    save_forecaster(ws, "demo")
    assert kind_of(owner("qa")) == CLASSIFIER and kind_of(owner("demo")) == FORECASTER
    assert owner("demo").name == "demo"
    with pytest.raises(KeyError):
        owner("energy")


def test_a_forecaster_keeps_its_own_methods_and_loads_only_approved_ones(ws):
    save_forecaster(ws, "demo")
    f = owner("demo")
    d = save_draft(f, "jitter", "scale noise", JITTER, "test")
    assert d["kind"] == FORECASTER and not d["problems"] and [m["id"] for m in drafts(f)] == [d["id"]]
    assert drafts(Forecaster(f.ws, "dem")) == []
    with pytest.raises(PermissionError):
        load(f, d["id"])
    decide(f, d["id"], APPROVED)
    assert set(load(f, d["id"])) == {"window", "after_step"}
    assert set(pickle.loads(pickle.dumps(Method(f, d["id"]))).hooks()) == {"window", "after_step"}


def test_the_app_lists_and_approves_the_methods_of_a_forecaster(ws):
    from fastapi.testclient import TestClient

    from byosm.app import create_app

    save_forecaster(ws, "demo")
    d = save_draft(owner("demo"), "jitter", "scale noise", JITTER, "test")
    with TestClient(create_app(ws, ""), base_url="http://127.0.0.1") as c:
        assert [m["kind"] for m in c.get("/api/specialists/demo/lab").json()] == [FORECASTER]
        assert c.post(f"/api/specialists/demo/lab/{d['id']}", json={"status": "approved"}).json()["status"] == APPROVED
        assert c.get("/api/specialists/nothing/lab").status_code == 404


def test_a_failing_rival_check_points_at_the_next_technique_then_at_the_lab(ws, monkeypatch):
    from byosm.series.health import health
    from byosm.series.steps import next_steps

    monkeypatch.setenv("BYOSM_CACHE", str(ws / "cache"))

    def rival(report):
        return next(c for c in health(report)["checks"] if c["id"] == "rival")

    behind = save_forecaster(ws, "demo", ours=0.9, rival=0.8)
    assert rival(behind)["techniques"] == ["Multi-rate forecasting"] and rival(behind)["args"][:2] == ["series", "run"]
    monkeypatch.setattr("byosm.series.prediction.next_technique", lambda report: None)
    assert rival(behind)["techniques"] == ["Lab method"] and rival(behind)["args"] == ["lab", "write", "demo"]
    steps = next_steps(ws, behind)
    assert steps[0]["args"] == ["lab", "write", "demo"] and steps[-1]["id"] == "pretraining-v2"
    ahead = save_forecaster(ws, "ahead", ours=0.7, rival=0.8)
    assert rival(ahead)["args"] is None and "techniques" not in rival(ahead)


def test_the_author_writes_a_forecaster_method_from_its_board_health_and_next_steps(ws, monkeypatch):
    from byosm.lab import author

    monkeypatch.setenv("BYOSM_CACHE", str(ws / "cache"))
    save_forecaster(ws, "demo")
    asked = []

    class Writer:
        def json(self, system, prompt, schema):
            asked.append((system, prompt))
            return {"name": "jitter", "hypothesis": "scale noise", "code": JITTER}

    monkeypatch.setattr("byosm.llm.get_llm", lambda role: Writer())
    d = author.write("demo", "decouple the horizons", lambda _: None)
    system, prompt = asked[0]
    assert d["kind"] == FORECASTER and not d["problems"] and drafts(owner("demo"))[0]["id"] == d["id"]
    assert "patch transformers" in system and "def window(" in system and "def rows(" not in system
    for part in ("datasets (a)", "Scores per dataset", "Timestral-5M", "Beats the best rival", "Try Multi-rate"):
        assert part in prompt, part
    assert "Methods written before" in prompt and prompt.endswith("The owner asks: decouple the horizons")
