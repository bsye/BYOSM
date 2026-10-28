import os
import time

from conftest import promoted_specialist
from fastapi.testclient import TestClient

from byosm.app import create_app, jobs
from byosm.measure import conduct
from byosm.measure.conduct import MAX_JOBS, Conductor

TIMEOUT_S = 30
POLL_S = 0.05
HARMLESS = "print('measured')"


def test_measure_lists_the_missing_measurements_and_runs_them_as_jobs(ws, monkeypatch):
    monkeypatch.setattr(jobs, "ENTRY", HARMLESS)
    monkeypatch.setattr(jobs, "job_env", lambda ws, job: dict(os.environ))
    monkeypatch.setattr(conduct, "TICK_S", POLL_S)
    promoted_specialist()
    app = create_app(ws, "")
    with TestClient(app, base_url="http://127.0.0.1") as client:
        plan = client.get("/api/measure").json()
        assert [(c["measurement"], c["status"]) for c in plan["cells"]] == [("overfit", "todo"), ("seeds", "todo")]
        assert plan["progress"]["todo"] == 2 and plan["error"] is None
        assert client.post("/api/measure", json={"names": ["nope"]}).status_code == 404
        started = client.post("/api/measure", json={}).json()
        assert {c["status"] for c in started["cells"]} <= {"pending", "running"}
        end = time.time() + TIMEOUT_S
        while {c["status"] for c in client.get("/api/measure").json()["cells"]} != {"failed"}:
            assert time.time() < end, "timed out"
            time.sleep(POLL_S)
        final = client.get("/api/measure").json()
        assert {c["note"] for c in final["cells"]} == {f"Still missing after {MAX_JOBS} jobs"}
        ran = client.get("/api/jobs").json()
        assert sorted(j["args"][0] for j in ran) == sorted(["overfit", "seeds"] * MAX_JOBS)
        assert {j["status"] for j in ran} == {"done"}
    conductor = next(h.__self__ for h in app.router.on_shutdown if isinstance(getattr(h, "__self__", None), Conductor))
    assert conductor.stopped
