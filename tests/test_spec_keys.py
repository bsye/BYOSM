import pytest

from byosm.llm import PROVIDERS, get_llm
from byosm.spec import Axis, DataPlan, Label, TaskSpec

SYNTH_EDITS = {
    "hard_cases": ["a confuser"],
    "diversity_axes": [Axis(name="tone", values=["calm", "angry"])],
    "languages": ["en", "it"],
    "summary": "another summary",
    "input_description": "chat messages",
    "name": "other",
}


def _spec(synthetic: bool, **kw) -> TaskSpec:
    labels = [Label(name="a", definition="x"), Label(name="b", definition="y")]
    base = {"name": "t", "description": "d", "modality": "text", "labels": labels}
    return TaskSpec(**{**base, **kw}, data=DataPlan(synthetic=synthetic))


@pytest.mark.parametrize("field", sorted(SYNTH_EDITS))
def test_synthetic_key_tracks_generation_inputs(field):
    assert _spec(True).data_key() != _spec(True, **{field: SYNTH_EDITS[field]}).data_key()


@pytest.mark.parametrize("field", sorted(SYNTH_EDITS))
def test_real_data_key_ignores_generation_inputs(field):
    assert _spec(False).data_key() == _spec(False, **{field: SYNTH_EDITS[field]}).data_key()


class _Fake:
    def __init__(self, **conf):
        self.conf = conf


def test_role_entry_merges_over_default(monkeypatch):
    monkeypatch.setitem(PROVIDERS, "fake", _Fake)
    default = {"provider": "fake", "model": "base", "base_url": "http://local", "api_key": "k"}
    cfg = {"llm": {"default": default, "eval": {"model": "other"}}}
    assert get_llm("eval", cfg).conf == {"model": "other", "base_url": "http://local", "api_key": "k"}
    assert get_llm("judge", cfg).conf == {"model": "base", "base_url": "http://local", "api_key": "k"}


def test_role_with_other_provider_ignores_default(monkeypatch):
    monkeypatch.setitem(PROVIDERS, "fake", _Fake)
    monkeypatch.setitem(PROVIDERS, "other", _Fake)
    default = {"provider": "fake", "model": "base", "base_url": "http://local", "api_key": "k"}
    cfg = {"llm": {"default": default, "eval": {"provider": "other", "model": "remote"}}}
    assert get_llm("eval", cfg).conf == {"model": "remote"}
    cfg["llm"]["eval"] = {"provider": "fake", "model": "remote"}
    assert get_llm("eval", cfg).conf == {"model": "remote"}


def test_role_on_another_endpoint_ignores_default(monkeypatch):
    monkeypatch.setitem(PROVIDERS, "fake", _Fake)
    local = {
        "provider": "fake",
        "model": "q",
        "base_url": "http://local",
        "api_key": "ollama",
        "reasoning_effort": "none",
    }
    remote = {"provider": "fake", "model": "gpt", "base_url": "https://openrouter.ai/api/v1", "api_key_env": "OR_KEY"}
    cfg = {"llm": {"default": local, "baseline": remote}}
    assert get_llm("baseline", cfg).conf == {k: v for k, v in remote.items() if k != "provider"}


def test_role_with_own_credential_uses_only_its_own_connection(monkeypatch):
    monkeypatch.setitem(PROVIDERS, "fake", _Fake)
    default = {"provider": "fake", "model": "base", "base_url": "http://local", "api_key": "k"}
    cfg = {"llm": {"default": default, "eval": {"model": "m", "api_key_env": "MY_KEY"}}}
    assert get_llm("eval", cfg).conf == {"model": "m", "api_key_env": "MY_KEY"}
