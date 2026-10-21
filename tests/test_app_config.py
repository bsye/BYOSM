import yaml
from fastapi.testclient import TestClient

from byosm.app import create_app

AUTH = {"Authorization": "Bearer t"}


def test_the_config_is_shown_without_keys_and_saved_without_losing_them(ws):
    (ws / "byosm.yaml").write_text(yaml.safe_dump({"llm": {"default": {"provider": "fake", "api_key": "sk-secret"}}}))
    with TestClient(create_app(ws, "t"), base_url="http://127.0.0.1") as c:
        shown = c.get("/api/config", headers=AUTH).json()
        assert "sk-secret" not in str(shown) and "********" in shown["text"]
        edited = shown["text"].replace("provider: fake", "provider: fake\n    model: m")
        assert c.put("/api/config", headers=AUTH, json={"text": edited}).status_code == 200
        assert c.put("/api/config", headers=AUTH, json={"text": "- a list"}).status_code == 422
    saved = yaml.safe_load((ws / "byosm.yaml").read_text())["llm"]["default"]
    assert saved == {"provider": "fake", "api_key": "sk-secret", "model": "m"}


def test_a_job_can_choose_its_own_llm_for_every_role(ws, monkeypatch):
    import json

    from byosm.config import LLM_OVERRIDE_ENV, load_config
    from byosm.llm import role_config

    chosen = {"provider": "claude-code", "model": "claude-haiku-5-5"}
    monkeypatch.setenv(LLM_OVERRIDE_ENV, json.dumps(chosen))
    assert role_config(load_config(ws)["llm"], "spec") == chosen
