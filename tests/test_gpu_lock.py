import threading
import time

from byosm.train.gpu import gpu_slot


def test_with_one_slot_only_one_job_trains_on_the_gpu_at_a_time(ws):
    import yaml

    cfg = yaml.safe_load((ws / "byosm.yaml").read_text())
    (ws / "byosm.yaml").write_text(yaml.safe_dump({**cfg, "gpu_jobs": 1}))
    order = []

    def job(name):
        with gpu_slot("mps", log=lambda _: None):
            order.append(f"{name} in")
            time.sleep(0.3)
            order.append(f"{name} out")

    import byosm.train.gpu as gpu

    gpu.WAIT_S = 0.05
    threads = [threading.Thread(target=job, args=(n,)) for n in "ab"]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert order[0].endswith("in") and order[1] == order[0].replace("in", "out")


def test_the_cpu_needs_no_slot(ws):
    with gpu_slot("cpu", log=lambda _: None), gpu_slot("cpu", log=lambda _: None):
        pass


def test_slots_follow_cores_and_memory_up_to_two(monkeypatch):
    import byosm.train.gpu as gpu

    class Mem:
        total = 64 * 1024**3

    monkeypatch.setattr("psutil.virtual_memory", lambda: Mem)
    monkeypatch.setattr("os.cpu_count", lambda: 18)
    assert gpu.auto_slots() == 2
    monkeypatch.setattr("os.cpu_count", lambda: 8)
    assert gpu.auto_slots() == 1
    Mem.total = 16 * 1024**3
    monkeypatch.setattr("os.cpu_count", lambda: 64)
    assert gpu.auto_slots() == 1
