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
