import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from byosm.app import create_app
from byosm.app.server import UNAUTHORIZED_CLOSE, allowed_origin

EVIL = {"Origin": "https://evil.com"}
FORBIDDEN = 403
EVENTS = "ws://127.0.0.1/api/events"


@pytest.fixture
def client(ws):
    with TestClient(create_app(ws, "", hosts=["box.lan"]), base_url="http://127.0.0.1") as c:
        yield c


def test_only_local_and_configured_origins_are_allowed():
    assert allowed_origin("", [])
    assert allowed_origin("http://localhost:5173", [])
    assert allowed_origin("http://[::1]:8765", [])
    assert allowed_origin("http://box.lan:8765", ["box.lan"])
    assert not allowed_origin("https://evil.com", [])
    assert not allowed_origin("null", [])
    assert not allowed_origin("http://127.0.0.1.evil.com", [])


def test_a_foreign_site_cannot_call_the_api_without_a_token(client):
    assert client.get("/api/jobs").status_code == 200
    assert client.get("/api/jobs", headers={"Origin": "http://localhost:5173"}).status_code == 200
    assert client.get("/api/jobs", headers=EVIL).status_code == FORBIDDEN
    assert client.post("/api/jobs/x/cancel", headers=EVIL).status_code == FORBIDDEN


def test_a_foreign_site_cannot_open_the_event_stream(client):
    with pytest.raises(WebSocketDisconnect) as closed, client.websocket_connect(EVENTS, headers=EVIL):
        pass
    assert closed.value.code == UNAUTHORIZED_CLOSE
    with client.websocket_connect(EVENTS, headers={"Origin": "http://127.0.0.1:8765"}):
        pass
