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
    llm._cache.clear()
    cfg = {"llm": {"default": {"provider": "slow", "base_url": "http://localhost:11434/v1", "model": "m"}}}
    model = llm.get_llm("default", cfg)
    threads = [threading.Thread(target=model.json, args=("s", "p", {})) for _ in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert _Slow.peak == 1
