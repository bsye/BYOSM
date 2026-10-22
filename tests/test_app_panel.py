import json

import pytest
from fastapi.testclient import TestClient

from byosm.app import create_app
from byosm.data.panel import PANEL_FILE
from byosm.data.review import reviewed_labels, write_review
from byosm.project import Project
from byosm.spec import Label, TaskSpec

TOKEN = "t0ken"
AUTH = {"Authorization": f"Bearer {TOKEN}"}
URL = "/api/specialists/topic/panel"
SUMMARY = {"teachers": {"llm": 0.8, "jev": 0.7}, "panel_on_labelled": 0.85, "n_labelled": 10, "all_agree": 0.5}


@pytest.fixture
def client(ws):
    spec = TaskSpec(
        name="topic",
        description="topic",
        modality="text",
        labels=[Label(name=n, definition=n) for n in ("food", "trip")],
    )
    spec.data.teachers = ["llm", "jev"]
    p = Project("topic", ws)
    p.root.mkdir(parents=True)
    p.save_spec(spec)
    dd = p.data_dir_for(spec)
    dd.mkdir(parents=True)
    (dd / PANEL_FILE).write_text(json.dumps(SUMMARY))
    write_review(dd, [{"input": "pasta trip", "panel": "food", "sure": 0.51, "llm": "food", "jev": "trip"}])
    with TestClient(create_app(ws, TOKEN), base_url="http://127.0.0.1") as c:
        yield c


def test_the_panel_needs_the_token_and_a_known_specialist(client):
    assert client.get(URL).status_code == 401
    assert client.put(f"{URL}/review", json={"input": "x", "label": "food"}).status_code == 401
    assert client.get("/api/specialists/missing/panel", headers=AUTH).status_code == 404


def test_the_panel_shows_teacher_accuracy_and_the_least_sure_rows(client):
    got = client.get(URL, headers=AUTH).json()
    assert got["teachers"] == ["llm", "jev"] and got["labels"] == ["food", "trip"] and got["summary"] == SUMMARY
    assert got["review"] == [
        {
            "input": "pasta trip",
            "panel": "food",
            "sure": 0.51,
            "votes": {"llm": "food", "jev": "trip"},
            "reviewed": None,
        }
    ]


def test_a_review_is_saved_replaced_and_shown_on_its_row(client, ws):
    review = f"{URL}/review"
    assert client.put(review, headers=AUTH, json={"input": "pasta trip", "label": "nope"}).status_code == 422
    assert client.put(review, headers=AUTH, json={"input": "pasta trip", "label": "food"}).json()["label"] == "food"
    assert client.put(review, headers=AUTH, json={"input": "pasta trip", "label": "trip"}).status_code == 200
    got = client.get(URL, headers=AUTH).json()
    assert got["review"][0]["reviewed"] == "trip" and got["n_reviewed"] == 1
    assert reviewed_labels(Project("topic", ws).reviewed_path) == {"pasta trip": "trip"}
