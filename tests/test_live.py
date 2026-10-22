import jwt
import pytest
import yaml
from fastapi.testclient import TestClient

from byosm import live
from byosm.app import create_app
from byosm.app.live_routes import reachable
from byosm.config import Project

AUTH = {"Authorization": "Bearer t"}


def test_settings_fall_back_to_the_livekit_dev_server_and_read_the_workspace_env(ws, monkeypatch):
    for k in (live.URL_KEY, live.API_KEY_KEY, live.SECRET_KEY, live.PUBLIC_URL_KEY):
        monkeypatch.delenv(k, raising=False)
    assert live.settings(ws) == {"url": live.DEV_URL, "public_url": live.DEV_URL, "key": "devkey", "secret": "secret"}
    (ws / ".env").write_text("LIVEKIT_URL=ws://lk:7890\nLIVEKIT_API_KEY=k\nLIVEKIT_API_SECRET=s" + "x" * 32 + "\n")
    assert live.settings(ws)["url"] == "ws://lk:7890"
    claims = jwt.decode(live.token("you", "room-1", ws), "s" + "x" * 32, algorithms=["HS256"])
    assert (claims["sub"], claims["video"]["room"], claims["video"]["roomJoin"]) == ("you", "room-1", True)


def test_a_closed_port_is_not_reachable():
    assert not reachable("ws://127.0.0.1:9")


@pytest.fixture
def audio(ws):
    p = Project("mic").ensure()
    spec = {"name": "mic", "description": "speech", "modality": "audio", "task_type": "frame"}
    p.spec_path.write_text(yaml.safe_dump({**spec, "labels": [{"name": "speech", "definition": "talking"}]}))
    return p


def test_the_live_endpoint_needs_an_export_and_a_livekit_server(audio, ws, monkeypatch):
    monkeypatch.setenv(live.URL_KEY, "ws://127.0.0.1:9")
    with TestClient(create_app(ws, "t"), base_url="http://127.0.0.1") as c:
        assert "export first" in c.post("/api/specialists/mic/live", headers=AUTH).json()["detail"]
        (audio.root / "export").mkdir()
        (audio.root / "export" / "manifest.json").write_text("{}")
        r = c.post("/api/specialists/mic/live", headers=AUTH)
        assert r.status_code == 503 and "livekit-server --dev" in r.json()["detail"]
        assert c.post("/api/specialists/none/live", headers=AUTH).status_code == 404
