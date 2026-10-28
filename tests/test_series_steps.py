from conftest import save_forecaster
from fastapi.testclient import TestClient

from byosm.app import create_app

AUTH = {"Authorization": "Bearer t"}


def test_a_forecaster_behind_its_rival_lists_the_fix_on_its_page_and_on_next_steps(ws, monkeypatch):
    monkeypatch.setattr("byosm.series.published.scores", lambda config, naive_mase: {})
    save_forecaster(ws, "demo", ours=0.95, rival=0.8)
    with TestClient(create_app(ws, "t"), base_url="http://127.0.0.1") as c:
        steps = c.get("/api/next-steps/demo", headers=AUTH).json()
        assert c.get("/api/series/demo", headers=AUTH).json()["next"] == steps
        rival = next(s for s in steps if s["id"] == "rival")
        assert rival["why"].startswith("Beats the best rival is not ok:") and rival["args"][:3] == [
            "series",
            "run",
            "demo",
        ]
        assert c.get("/api/next-steps/nothing", headers=AUTH).status_code == 404
