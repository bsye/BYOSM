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

    own = {"llm": {"default": {"provider": "fake"}, "spec": {"provider": "anthropic", "model": "claude-opus-5-5"}}}
    (ws / "byosm.yaml").write_text(yaml.safe_dump(own))
    chosen = {"provider": "claude-code", "model": "claude-haiku-5-5"}
    monkeypatch.setenv(LLM_OVERRIDE_ENV, json.dumps(chosen))
    assert role_config(load_config(ws)["llm"], "spec") == chosen


def test_the_default_llm_can_be_set_from_a_provider_choice(ws):
    with TestClient(create_app(ws, "t"), base_url="http://127.0.0.1") as c:
        chosen = {"provider": "claude-code", "model": "claude-haiku-5-5", "base_url": None}
        assert c.put("/api/config/default-llm", headers=AUTH, json={"config": chosen}).status_code == 200
    saved = yaml.safe_load((ws / "byosm.yaml").read_text())["llm"]["default"]
    assert saved == {"provider": "claude-code", "model": "claude-haiku-5-5"}


def test_the_hugging_face_user_comes_from_the_workspace_token_and_is_none_when_logged_out(ws, monkeypatch):
    import huggingface_hub

    from byosm.app import config_routes

    config_routes._hub_user.cache_clear()
    monkeypatch.delenv("HF_TOKEN", raising=False)
    (ws / ".env").write_text("HF_TOKEN=hf_test\n")
    seen = []

    def whoami(token=None):
        seen.append(token)
        if token != "hf_test":
            raise OSError("not logged in")
        return {"name": "someone"}

    monkeypatch.setattr(huggingface_hub, "whoami", whoami)
    with TestClient(create_app(ws, "t"), base_url="http://127.0.0.1") as c:
        assert c.get("/api/hub/user", headers=AUTH).json() == {"user": "someone"}
        (ws / ".env").write_text("HF_TOKEN=other\n")
        assert c.get("/api/hub/user", headers=AUTH).json() == {"user": None}
    assert seen == ["hf_test", "other"]
