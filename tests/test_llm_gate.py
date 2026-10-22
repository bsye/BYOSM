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


def test_a_local_llm_server_gets_one_call_at_a_time():
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
