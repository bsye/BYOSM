import pytest
from conftest import save_forecaster
from fastapi.testclient import TestClient
from test_lab import FOCAL, _project
from test_lab_forecaster import JITTER

from byosm.app import create_app
from byosm.app.hub import Hub
from byosm.app.jobs import Runner
from byosm.auto import ready
from byosm.db.workspace import save_workspace_report
from byosm.lab.drafts import APPROVED, decide, save_draft, update
from byosm.lab.owners import Forecaster
from byosm.lab.steps import lab_steps

AUTH = {"Authorization": "Bearer t"}
OTHER = FOCAL.replace("i % 4", "i % 5")
BAD = "import os\ndef rows(t, r):\n    return t"


def test_a_pending_method_waits_for_a_person_and_an_approved_one_becomes_a_run(ws):
    p = _project()
    waiting = save_draft(p, "focal", "Focus on hard rows. Then more.", FOCAL, "test")
    approved = save_draft(p, "curriculum", "Hard rows last.", OTHER, "test")
    save_draft(p, "bad", "x", BAD, "test")
    decide(p, approved["id"], APPROVED)
    steps = {s["id"]: s for s in lab_steps(p)}
    assert set(steps) == {f"approve-lab-{waiting['id']}", f"run-lab-{approved['id']}"}
    assert steps[f"approve-lab-{waiting['id']}"] | {"why": ""} == {
        "id": f"approve-lab-{waiting['id']}",
        "title": "Approve lab method focal",
        "why": "",
        "args": [],
        "approve": waiting["id"],
    }
    assert steps[f"approve-lab-{waiting['id']}"]["why"].endswith("Focus on hard rows.")
    assert steps[f"run-lab-{approved['id']}"]["args"] == ["lab", "run", "tc", approved["id"]]
    update(p, approved["id"], result={"experiment": "v2", "won": False, "calib": 0.5})
    assert [s["id"] for s in lab_steps(p)] == [f"approve-lab-{waiting['id']}"]


def test_the_ready_loop_never_takes_an_approval_step(monkeypatch):
    approval = {"id": "approve-lab-a", "title": "Approve", "why": "", "args": [], "approve": "a", "gain": 9}
    run = {"id": "run-lab-b", "title": "Run", "why": "", "args": ["lab", "run", "tc", "b"], "gain": 0}
    monkeypatch.setattr("byosm.next_steps.suggestions", lambda name: [approval, run])
    assert ready._next("tc", set()) == run


def test_approving_lab_code_cannot_be_a_job(ws):
    with pytest.raises(ValueError):
        Runner(ws, Hub(), parallel=1).submit(["lab", "approve", "tc", "abc"])


def test_a_forecaster_lists_its_pending_method_and_a_proposal_lists_the_methods_it_produced(ws, monkeypatch):
    monkeypatch.setattr("byosm.series.published.scores", lambda config, naive_mase: {})
    save_forecaster(ws, "demo", ours=0.95, rival=0.8)
    paper = "https://arxiv.org/abs/2"
    d = save_draft(Forecaster(ws, "demo"), "jitter", "Scale noise.", JITTER, "test")
    update(Forecaster(ws, "demo"), d["id"], paper=paper)
    proposal = {"id": "s1", "title": "DHF", "paper": paper, "effect": "e", "kinds": ["series"], "sent": ["demo"]}
    save_workspace_report(ws, "scout", {"last_run": 1, "seen": [], "proposals": [proposal]})
    with TestClient(create_app(ws, "t"), base_url="http://127.0.0.1") as c:
        steps = c.get("/api/next-steps/demo", headers=AUTH).json()
        assert {"id": f"approve-lab-{d['id']}", "approve": d["id"]}.items() <= next(
            s for s in steps if s.get("approve")
        ).items()
        listed = c.get("/api/scout", headers=AUTH).json()["proposals"][0]["methods"]
        assert [(m["owner"], m["id"], m["status"]) for m in listed] == [("demo", d["id"], "pending")]
