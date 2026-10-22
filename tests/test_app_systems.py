import yaml
from fastapi.testclient import TestClient

from byosm.app import create_app
from byosm.app.teams import members

AUTH = {"Authorization": "Bearer t"}


def test_typing_to_a_system_that_is_not_exported_says_how_to_fix_it(ws):
    with TestClient(create_app(ws, "t"), base_url="http://127.0.0.1") as c:
        r = c.post("/api/systems/none/type", headers=AUTH, json={"text": "wake me up"})
        assert r.status_code == 404 and "system build" in r.json()["detail"]
        assert c.post("/api/systems/none/type", json={"text": "x"}).status_code == 401


def _team(ws, name, specialists):
    d = ws / "systems" / name
    d.mkdir(parents=True)
    source = {"dataset": "org/data", "input_column": "text", "label_column": "label"}
    roles = ["router", "tagger"]
    nodes = {s: {"specialist": s, "role": roles[i], "task": s, "source": source} for i, s in enumerate(specialists)}
    (d / "system.yaml").write_text(yaml.safe_dump({"name": name, "description": name, "nodes": nodes}))


def test_only_teams_of_several_models_fold_their_members_into_one_entry(ws):
    _team(ws, "solo", ["router"])
    _team(ws, "pair", ["picker", "filler"])
    assert members(ws) == {"picker": "pair", "filler": "pair"}
