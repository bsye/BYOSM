from conftest import save_forecaster
from fastapi.testclient import TestClient
from test_lab_forecaster import JITTER

from byosm.app import create_app
from byosm.db.workspace import save_workspace_report
from byosm.lab.drafts import save_draft, update
from byosm.lab.owners import Forecaster
from byosm.scout.tried import lab_families, tried

AUTH = {"Authorization": "Bearer t"}
PAPER = "https://arxiv.org/abs/2610.00002"


def _proposal(ws, ran):
    save_forecaster(ws, "demo", ours=0.95, rival=0.8)
    f = Forecaster(ws, "demo")
    d = save_draft(f, "jitter", "Scale noise.", JITTER, "test")
    update(f, d["id"], paper=PAPER, **({"result": {"experiment": "v2", "won": True, "calib": 0.9}} if ran else {}))
    p = {"id": "s1", "title": "DHF", "paper": PAPER, "effect": "e", "kinds": ["series"], "sent": ["demo"], "found": 1.0}
    save_workspace_report(ws, "scout", {"last_run": 1, "seen": [], "proposals": [p]})
    return p


def test_a_proposal_whose_methods_have_not_run_stays_proposed(ws, monkeypatch):
    monkeypatch.setattr("byosm.series.published.scores", lambda config, naive_mase: {})
    p = _proposal(ws, ran=False)
    assert not tried(ws, p) and lab_families(ws) == []


def test_a_tried_proposal_joins_the_catalogue_and_a_forecaster_promotion_is_not_a_sure_lead(ws, monkeypatch):
    monkeypatch.setattr("byosm.series.published.scores", lambda config, naive_mase: {})
    p = _proposal(ws, ran=True)
    assert tried(ws, p)
    [row] = lab_families(ws)
    assert row["status"] == "lab" and row["tries"] == 1 and row["specialists"] == 1 and row["wins"] == 0
    assert row["outcomes"] == [{"owner": "demo", "method": "jitter", "result": "promoted"}]
    with TestClient(create_app(ws, "t"), base_url="http://127.0.0.1") as c:
        assert c.get("/api/scout", headers=AUTH).json()["proposals"][0]["tried"] is True
        assert [t["id"] for t in c.get("/api/techniques", headers=AUTH).json() if t["status"] == "lab"] == ["scout-s1"]
