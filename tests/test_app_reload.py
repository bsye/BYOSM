from fastapi.testclient import TestClient

from byosm.app.factory import from_env, settings_env


def test_a_reloading_app_is_rebuilt_from_the_settings_the_command_gave_it(ws, monkeypatch):
    for k, v in settings_env(ws, "t", ["box.local"], 2, 8765).items():
        monkeypatch.setenv(k, v)
    with TestClient(from_env(), base_url="http://box.local") as c:
        assert c.get("/api/specialists", headers={"Authorization": "Bearer t"}).status_code == 200
        assert c.get("/api/specialists").status_code == 401
