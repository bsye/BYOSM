import io
import json
import subprocess

import pytest

from byosm.llm import detect as det

LISTINGS = {
    "http://localhost:11434/api/tags": {"models": [{"name": "qwen3.8:27b"}]},
    "http://localhost:1234/v1/models": {"data": [{"id": "local-model"}]},
    "https://api.openai.com/v1/models": {"data": [{"id": "whisper-1"}, {"id": "gpt-a"}, {"id": "gpt-b"}]},
    "https://api.anthropic.com/v1/models": {"data": [{"id": "claude-x"}]},
}
SECRETS = {"OPENAI_API_KEY": "sk-openai-secret", "ANTHROPIC_API_KEY": "sk-ant-secret"}


def _machine(monkeypatch, servers: set[str], binaries: set[str], secrets: dict[str, str]) -> list:
    requests = []

    def urlopen(request, timeout):
        requests.append(request)
        if request.full_url not in servers:
            raise OSError("connection refused")
        return io.BytesIO(json.dumps(LISTINGS[request.full_url]).encode())

    monkeypatch.setattr(det.urllib.request, "urlopen", urlopen)
    monkeypatch.setattr(det.shutil, "which", lambda b: f"/bin/{b}" if b in binaries else None)
    monkeypatch.setattr(
        det.subprocess, "run", lambda cmd, **_: subprocess.CompletedProcess(cmd, 0, stdout="1.0\n", stderr="")
    )
    monkeypatch.setattr(det, "secret", secrets.get)
    return requests


def _by_id() -> dict:
    return {p["id"]: p for p in det.detect()["providers"]}


def test_every_known_provider_is_listed_with_one_shape(monkeypatch):
    _machine(monkeypatch, set(), set(), {})
    found = det.detect()["providers"]
    assert [p["id"] for p in found] == ["claude-code", "codex", "ollama", "lmstudio", "openai", "anthropic"]
    labels = {p["label"] for p in found}
    assert labels == {"Claude Code", "Codex", "Ollama", "LM Studio", "OpenAI API", "Anthropic API"}
    for p in found:
        assert not p["available"] and p["config"] is None and p["models"] == [] and p["hint"]
        assert p["kind"] in {"local agent", "local model", "hosted"}


def test_local_servers_give_openai_role_configs(monkeypatch):
    _machine(monkeypatch, set(LISTINGS), set(), {})
    found = _by_id()
    assert found["ollama"]["config"] == {
        "provider": "openai",
        "base_url": "http://localhost:11434/v1",
        "api_key": "ollama",
        "model": "qwen3.8:27b",
    }
    lmstudio = found["lmstudio"]
    assert lmstudio["available"] and lmstudio["models"] == ["local-model"] and "hint" not in lmstudio
    assert lmstudio["config"]["base_url"] == "http://localhost:1234/v1"
    assert lmstudio["config"]["provider"] == "openai" and lmstudio["config"]["model"] == "local-model"


def test_installed_agents_report_path_and_version(monkeypatch):
    _machine(monkeypatch, set(), {"claude", "codex"}, {})
    found = _by_id()
    assert found["claude-code"]["config"]["provider"] == "claude-code"
    assert found["codex"]["config"] == {"provider": "codex", "model": None}
    assert found["codex"]["path"] == "/bin/codex" and found["codex"]["version"] == "1.0"


def test_hosted_apis_need_a_key_and_never_return_it(monkeypatch):
    requests = _machine(monkeypatch, set(LISTINGS), set(), SECRETS)
    found = _by_id()
    assert found["openai"]["models"] == ["gpt-b", "gpt-a"]
    assert found["openai"]["config"] == {"provider": "openai", "api_key_env": "OPENAI_API_KEY", "model": "gpt-b"}
    assert found["anthropic"]["config"]["api_key_env"] == "ANTHROPIC_API_KEY"
    assert found["anthropic"]["models"] == ["claude-x"]
    assert all(secret not in json.dumps(found) for secret in SECRETS.values())
    sent = {r.full_url: r for r in requests}
    assert sent["https://api.anthropic.com/v1/models"].get_header("X-api-key") == "sk-ant-secret"


@pytest.mark.parametrize("pid", ["openai", "anthropic"])
def test_hosted_api_with_a_key_but_no_listing_stays_available(monkeypatch, pid):
    _machine(monkeypatch, set(), set(), SECRETS)
    assert _by_id()[pid]["available"]
