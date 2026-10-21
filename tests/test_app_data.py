import json

import pytest
from fastapi.testclient import TestClient

from byosm.app import create_app, data_routes
from byosm.data.request import REQUEST_DIR
from byosm.spec import Label, TaskSpec

TOKEN = "t0ken"
AUTH = {"Authorization": f"Bearer {TOKEN}"}
TABLE = b"text,label,split\nhello,a,\n"


@pytest.fixture
def client(ws):
    spec = TaskSpec(name="q", description="is it a question", modality="text", labels=[Label(name="a", definition="")])
    (ws / "specialists" / "q").mkdir(parents=True)
    spec.save(ws / "specialists" / "q" / "spec.yaml")
    with TestClient(create_app(ws, TOKEN), base_url="http://127.0.0.1") as c:
        yield c


def _request(ws):
    out = ws / "specialists" / "q" / REQUEST_DIR
    out.mkdir(parents=True)
    (out / "request.json").write_text(json.dumps({"reason": "r", "counts": {"a": {"train_min": 50}}}))
    (out / "template.csv").write_text("text,label,split\n")
    return out


def test_data_routes_need_the_token(client):
    assert client.get("/api/specialists/q/request").status_code == 401
    assert client.post("/api/specialists/q/uploads", files={"file": ("t.csv", TABLE)}).status_code == 401


def test_the_request_and_its_templates_are_served(client, ws):
    empty = client.get("/api/specialists/q/request", headers=AUTH).json()
    assert empty == {"request": None, "templates": [], "uploads": []}
    assert client.get("/api/specialists/q/request/template.csv", headers=AUTH).status_code == 404
    _request(ws)
    got = client.get("/api/specialists/q/request", headers=AUTH).json()
    assert got["request"]["counts"]["a"]["train_min"] == 50 and got["templates"] == ["csv"]
    csv = client.get("/api/specialists/q/request/template.csv", headers=AUTH)
    assert (
        csv.status_code == 200
        and csv.text == "text,label,split\n"
        and "q-template.csv" in csv.headers["content-disposition"]
    )
    assert client.get("/api/specialists/q/request/template.pdf", headers=AUTH).status_code == 422
    assert client.get("/api/specialists/missing/request", headers=AUTH).status_code == 404


def test_a_filled_table_is_uploaded_into_the_specialist(client, ws):
    up = client.post("/api/specialists/q/uploads", headers=AUTH, files={"file": ("../../my table.csv", TABLE)})
    assert up.json() == {"path": "specialists/q/uploads/my_table.csv", "bytes": len(TABLE)}
    assert (ws / up.json()["path"]).read_bytes() == TABLE
    listed = client.get("/api/specialists/q/request", headers=AUTH).json()["uploads"]
    assert listed == ["specialists/q/uploads/my_table.csv"]


@pytest.mark.parametrize("filename", ["notes.txt", "..", ".hidden.csv", "x.csv.sh", ""])
def test_uploads_that_are_not_tables_are_refused(client, ws, filename):
    up = client.post("/api/specialists/q/uploads", headers=AUTH, files={"file": (filename, TABLE)})
    assert up.status_code == 422
    assert not (ws / "specialists" / "q" / "uploads").exists()


def test_an_upload_over_the_size_limit_is_refused(client, ws, monkeypatch):
    monkeypatch.setattr(data_routes, "MAX_UPLOAD_BYTES", len(TABLE) - 1)
    up = client.post("/api/specialists/q/uploads", headers=AUTH, files={"file": ("t.csv", TABLE)})
    assert up.status_code == 413 and not (ws / "specialists" / "q" / "uploads").exists()


def test_the_check_report_is_served_once_written(client, ws):
    assert client.get("/api/specialists/q/check", headers=AUTH).json() == {"report": None}
    (_request(ws) / "report.json").write_text(json.dumps({"verdict": "ready"}))
    assert client.get("/api/specialists/q/check", headers=AUTH).json()["report"]["verdict"] == "ready"
