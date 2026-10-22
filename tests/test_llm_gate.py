import threading
import time

from byosm import llm


class _Slow:
    name = "slow"
    active = 0
    peak = 0
    lock = threading.Lock()

    def __init__(self, **_):
        pass

    def json(self, system, prompt, schema):
        with _Slow.lock:
            _Slow.active += 1
            _Slow.peak = max(_Slow.peak, _Slow.active)
        time.sleep(0.05)
        with _Slow.lock:
            _Slow.active -= 1
        return {}


def test_a_local_llm_server_gets_as_many_calls_as_free_memory_allows(monkeypatch):
    from byosm import resources

    monkeypatch.setattr(resources, "free_gb", lambda: resources.LOCAL_CALL_GB * 1.5)
    llm.register_provider("slow", _Slow)
    llm.clear_cache()
    cfg = {"llm": {"default": {"provider": "slow", "base_url": "http://localhost:11434/v1", "model": "m"}}}
    model = llm.get_llm("default", cfg)
    threads = [threading.Thread(target=model.json, args=("s", "p", {})) for _ in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert _Slow.peak == 1


def test_roles_sharing_a_model_keep_their_own_name_and_one_gate():
    llm.register_provider("slow", _Slow)
    llm.clear_cache()
    cfg = {"llm": {"default": {"provider": "slow", "base_url": "http://localhost:11434/v1", "model": "m"}}}
    spec, test = llm.get_llm("spec", cfg), llm.get_llm("eval", cfg)
    assert (spec.role, test.role) == ("spec", "eval")
    assert spec.gate is test.gate


def test_a_remote_llm_has_no_call_limit_and_free_memory_sets_the_local_one(monkeypatch):
    from byosm import resources

    llm.clear_cache()
    remote = {"llm": {"default": {"provider": "slow", "base_url": "https://api.example.com/v1", "model": "m"}}}
    assert llm.get_llm("default", remote).limit is None
    monkeypatch.setattr(resources, "free_gb", lambda: resources.LOCAL_CALL_GB * 4.2)
    local = {"llm": {"default": {"provider": "slow", "base_url": "http://localhost:1234/v1", "model": "m2"}}}
    assert llm.get_llm("default", local).limit == 4
