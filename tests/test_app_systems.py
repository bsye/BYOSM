from fastapi.testclient import TestClient

from byosm.app import create_app

AUTH = {"Authorization": "Bearer t"}


def test_typing_to_a_system_that_is_not_exported_says_how_to_fix_it(ws):
    with TestClient(create_app(ws, "t"), base_url="http://127.0.0.1") as c:
        r = c.post("/api/systems/none/type", headers=AUTH, json={"text": "wake me up"})
        assert r.status_code == 404 and "system build" in r.json()["detail"]
        assert c.post("/api/systems/none/type", json={"text": "x"}).status_code == 401
