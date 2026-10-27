from fastapi.testclient import TestClient

from byosm.app.server import create_app
from byosm.project import Project
from byosm.spec import Label, TaskSpec

TOKEN = "t0ken"
AUTH = {"Authorization": f"Bearer {TOKEN}"}


def _specialist(ws, name="kept"):
    p = Project(name).ensure()
    p.save_spec(TaskSpec(name=name, description="", modality="text", labels=[Label(name="a", definition="")]))
    return p


def test_an_archived_specialist_is_flagged_in_the_list_and_restored_on_request(ws):
    from byosm.app.catalog import specialists

    _specialist(ws)
    with TestClient(create_app(ws, TOKEN), base_url="http://127.0.0.1") as client:
        on = client.post("/api/specialists/kept/archive", json={"archived": True}, headers=AUTH)
        assert on.json()["archived"] and next(s for s in specialists(ws) if s["name"] == "kept")["archived"]
        off = client.post("/api/specialists/kept/archive", json={"archived": False}, headers=AUTH)
        assert not off.json()["archived"] and not next(s for s in specialists(ws) if s["name"] == "kept")["archived"]


def test_a_card_edit_that_adds_a_number_is_refused(monkeypatch):
    import byosm.llm
    from byosm.hub.preview import edit

    class Writer:
        def __init__(self, reply):
            self.reply = reply

        def json(self, system, prompt, schema):
            return {"markdown": self.reply}

    card = "# Tool choice\n\nIt scores **0.962** at 11.2M parameters."
    monkeypatch.setattr(
        byosm.llm, "get_llm", lambda role, cfg=None: Writer(card.replace("It scores", "Shorter: it scores"))
    )
    assert edit(card, "shorter")["refused"] is None
    monkeypatch.setattr(byosm.llm, "get_llm", lambda role, cfg=None: Writer(card + " Beats the LLM by 0.4."))
    out = edit(card, "say it beats the LLM")
    assert out["card"] == card and "0.4" in out["refused"]
