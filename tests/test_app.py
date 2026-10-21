import json
import time

import pytest
from fastapi.testclient import TestClient

from byosm.app import create_app
from byosm.app.jobs import LOST_LINE
from byosm.app.store import JobStore
from byosm.spec import Label, TaskSpec

TOKEN = "t0ken"
AUTH = {"Authorization": f"Bearer {TOKEN}"}
POLL_SECONDS = 0.05


@pytest.fixture
def client(ws):
    spec = TaskSpec(name="q", description="is it a question", modality="text", labels=[Label(name="a", definition="")])
    (ws / "specialists" / "q").mkdir(parents=True)
    spec.save(ws / "specialists" / "q" / "spec.yaml")
    with TestClient(create_app(ws, TOKEN), base_url="http://127.0.0.1") as c:
        yield c


def test_the_api_needs_the_token(client):
    assert client.get("/api/specialists").status_code == 401
    assert client.get("/api/specialists", headers={"Authorization": "Bearer nope"}).status_code == 401


def test_specialists_are_listed_read_and_edited(client):
    listed = client.get("/api/specialists", headers=AUTH).json()
    assert [s["name"] for s in listed] == ["q"] and listed[0]["promoted"] is None
    spec = client.get("/api/specialists/q", headers=AUTH).json()["spec"]
    spec["description"] = "a new description"
    assert client.put("/api/specialists/q/spec", headers=AUTH, json=spec).json()["description"] == "a new description"
    assert client.get("/api/specialists/missing", headers=AUTH).status_code == 404
    assert client.delete("/api/specialists/q", headers=AUTH, params={"confirm": "x"}).status_code == 422


def test_the_specialist_detail_carries_the_overfitting_check(client, ws):
    assert client.get("/api/specialists/q", headers=AUTH).json()["overfit"] is None
    report = {"experiment": "v1", "verdict": "warn", "findings": []}
    (ws / "specialists" / "q" / "overfit.json").write_text(json.dumps(report))
    assert client.get("/api/specialists/q", headers=AUTH).json()["overfit"] == report


def test_keys_are_stored_in_the_workspace_and_never_returned(client, ws):
    assert client.put("/api/keys/JEV_API_KEY", headers=AUTH, json={"value": "abc"}).status_code == 200
    keys = {k["name"]: k for k in client.get("/api/keys", headers=AUTH).json()}
    assert keys["JEV_API_KEY"]["set"] and "abc" not in str(keys)
    assert (ws / ".env").read_text() == "JEV_API_KEY=abc\n" and (ws / ".env").stat().st_mode & 0o077 == 0
    assert client.put("/api/keys/bad name", headers=AUTH, json={"value": "x"}).status_code == 422


def test_a_job_runs_the_cli_and_streams_its_output(client):
    assert client.post("/api/jobs", headers=AUTH, json={"args": ["rm", "-rf"]}).status_code == 422
    job = client.post("/api/jobs", headers=AUTH, json={"args": ["suite", "--help"]}).json()
    with client.websocket_connect(f"ws://127.0.0.1/api/events?token={TOKEN}") as sock:
        status = ""
        while status not in ("done", "failed"):
            ev = sock.receive_json()
            if ev["type"] == "job" and ev["job"]["id"] == job["id"]:
                status = ev["job"]["status"]
    done = client.get(f"/api/jobs/{job['id']}", headers=AUTH).json()
    assert done["status"] == "done" and any("suite" in line for line in done["lines"])


def test_job_history_survives_a_restart(client, ws):
    job = client.post("/api/jobs", headers=AUTH, json={"args": ["suite", "--help"]}).json()
    runner = client.app.state.runner
    while runner.jobs[job["id"]].status != "done":
        time.sleep(POLL_SECONDS)
    done = client.get(f"/api/jobs/{job['id']}", headers=AUTH).json()
    JobStore(ws).save({**done, "id": "stale", "status": "running"})
    with TestClient(create_app(ws, TOKEN), base_url="http://127.0.0.1") as again:
        assert again.get(f"/api/jobs/{job['id']}", headers=AUTH).json()["lines"] == done["lines"]
        stale = again.get("/api/jobs/stale", headers=AUTH).json()
    assert stale["status"] == "failed" and stale["lines"] == [LOST_LINE]
    assert (ws / ".byosm" / "app.sqlite").exists()


def test_the_app_command_takes_a_fixed_token_and_extra_hosts(ws, monkeypatch):
    import uvicorn
    from typer.testing import CliRunner

    from byosm.cli import app

    served = {}
    monkeypatch.setattr(uvicorn, "run", lambda api, **kw: served.update(api=api, **kw))
    env = {"BYOSM_APP_TOKEN": TOKEN, "BYOSM_APP_HOSTS": "byosm.internal, app.example"}
    out = CliRunner().invoke(app, ["app", "--no-open", "--host", "0.0.0.0"], env=env)
    assert out.exit_code == 0 and TOKEN not in out.output and served["host"] == "0.0.0.0"
    for base, code in [("http://byosm.internal", 200), ("http://app.example", 200), ("http://evil.example", 400)]:
        with TestClient(served["api"], base_url=base) as c:
            assert c.get("/api/specialists", headers=AUTH).status_code == code


def test_a_second_app_on_the_same_workspace_is_refused(ws):
    import json
    import os

    from byosm.app.lock import LOCK_PATH, WorkspaceBusyError

    (ws / ".byosm").mkdir(exist_ok=True)
    (ws / LOCK_PATH).write_text(json.dumps({"pid": os.getppid(), "port": 1}))
    with pytest.raises(WorkspaceBusyError):
        create_app(ws, TOKEN)
