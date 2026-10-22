import yaml
from fastapi.testclient import TestClient

from byosm.app import create_app

AUTH = {"Authorization": "Bearer t"}


def test_usage_adds_up_tokens_per_job_task_and_model(ws):
    app = create_app(ws, "t")
    store = app.state.runner.store
    done = {"kind": "llm", "role": "generate", "model": "m", "state": "done", "at": 1.0, "seconds": 2.0}
    store.event("j1", {**done, "tokens_in": 100, "tokens_out": 20})
    store.event("j1", {**done, "tokens_in": 50, "tokens_out": 5})
    store.event("j1", {**done, "state": "waiting"})
    with TestClient(app, base_url="http://127.0.0.1") as c:
        (row,) = c.get("/api/llm/usage", headers=AUTH).json()
    assert row | {"last": None} == {
        "job": "j1",
        "role": "generate",
        "model": "m",
        "calls": 2,
        "failed": 0,
        "tokens_in": 150,
        "tokens_out": 25,
        "seconds": 4.0,
        "last": None,
        "counted": 2,
        "counted_seconds": 4.0,
    }


def test_effort_is_saved_under_each_providers_own_key_and_the_budget_round_trips(ws):
    (ws / "byosm.yaml").write_text(yaml.safe_dump({"llm": {"default": {"provider": "openai", "model": "q"}}}))
    with TestClient(create_app(ws, "t"), base_url="http://127.0.0.1") as c:
        c.put("/api/llm/roles/judge", headers=AUTH, json={"settings": {"effort": "low", "max_parallel": 2}})
        c.put("/api/llm/roles/spec", headers=AUTH, json={"settings": {"provider": "claude-code", "effort": "high"}})
        shown = c.put("/api/llm/budget", headers=AUTH, json={"tokens_per_job": 50000}).json()
        bad = c.put("/api/llm/roles/judge", headers=AUTH, json={"settings": {"temperature": 1}})
    saved = yaml.safe_load((ws / "byosm.yaml").read_text())
    assert saved["llm"]["judge"] == {"reasoning_effort": "low", "max_parallel": 2}
    assert saved["llm"]["spec"] == {"provider": "claude-code", "effort": "high"}
    assert saved["budget"] == {"tokens_per_job": 50000} and shown["budget"] == {"tokens_per_job": 50000}
    judge = next(r for r in shown["roles"] if r["role"] == "judge")
    assert (judge["effort"], judge["model"], judge["own"]) == ("low", "q", True)
    assert bad.status_code == 422
