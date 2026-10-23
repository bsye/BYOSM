import threading

import byosm.train.gpu as gpu


def _short_of_memory(monkeypatch):
    monkeypatch.setattr(gpu, "free_gb", lambda: 1.0)
    monkeypatch.setattr(gpu, "WAIT_S", 0.01)


def test_a_lone_training_starts_even_when_memory_is_short(ws, monkeypatch):
    _short_of_memory(monkeypatch)
    lines = []
    with gpu.gpu_slot("mps", log=lines.append):
        pass
    assert not any("waiting" in line for line in lines)


def test_a_second_training_waits_for_memory_while_another_holds_a_slot(ws, monkeypatch):
    import yaml

    cfg = yaml.safe_load((ws / "byosm.yaml").read_text())
    (ws / "byosm.yaml").write_text(yaml.safe_dump({**cfg, "gpu_jobs": 2}))
    _short_of_memory(monkeypatch)
    started, release, second = threading.Event(), threading.Event(), []

    def first():
        with gpu.gpu_slot("mps", log=lambda _: None):
            started.set()
            release.wait(5)

    holder = threading.Thread(target=first)
    holder.start()
    started.wait(5)

    def wait_then_train():
        with gpu.gpu_slot("mps", log=second.append):
            pass

    waiter = threading.Thread(target=wait_then_train)
    waiter.start()
    waiter.join(0.3)
    assert waiter.is_alive() and any("waiting" in line for line in second)
    release.set()
    holder.join(5)
    waiter.join(5)
    assert not waiter.is_alive()


def test_cuda_does_not_wait_on_host_memory(ws, monkeypatch):
    _short_of_memory(monkeypatch)
    monkeypatch.setattr(gpu, "busy_slots", lambda _: 1)
    lines = []
    with gpu.gpu_slot("cuda", log=lines.append):
        pass
    assert not any("waiting" in line for line in lines)


def test_the_memory_needed_never_exceeds_what_the_machine_can_give(monkeypatch):
    class Mem:
        total = 16 * 1024**3

    monkeypatch.setattr("psutil.virtual_memory", lambda: Mem)
    assert gpu.needed_gb() < 16
    Mem.total = 128 * 1024**3
    assert gpu.needed_gb() == gpu.GB_PER_TRAINING
