import time

import byosm.train.gpu as gpu
from byosm.train.gpu import gpu_slot, held_seconds


def test_time_holding_a_gpu_slot_is_added_up_and_the_cpu_adds_nothing(ws, monkeypatch):
    monkeypatch.setattr(gpu, "free_gb", lambda: 100.0)
    before = held_seconds()
    with gpu_slot("cpu", log=lambda _: None):
        time.sleep(0.05)
    assert held_seconds() == before
    with gpu_slot("mps", log=lambda _: None):
        time.sleep(0.2)
    assert held_seconds() - before >= 0.2
