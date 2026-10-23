import threading
import time

from byosm.train.gpu import gpu_slot


def test_with_one_slot_only_one_job_trains_on_the_gpu_at_a_time(ws, monkeypatch):
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

    monkeypatch.setattr(gpu, "WAIT_S", 0.05)
    monkeypatch.setattr(gpu, "_roomy", lambda: True)
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


def test_a_training_waits_while_memory_is_short(ws, monkeypatch):
    import byosm.train.gpu as gpu

    free = iter([2.0, 2.0, 40.0])
    monkeypatch.setattr(gpu, "free_gb", lambda: next(free))
    monkeypatch.setattr(gpu, "WAIT_S", 0.01)
    lines = []
    with gpu.gpu_slot("cuda", log=lines.append):
        pass
    assert any("waiting" in line for line in lines) and any("got the GPU" in line for line in lines)


def test_running_out_of_gpu_memory_halves_the_batch_until_it_fits():
    from byosm.spec import Label, TaskSpec
    from byosm.train.gpu import shrink_on_oom

    spec = TaskSpec(name="q", description="q", modality="text", labels=[Label(name="a", definition="")])
    tried = []

    def train(s):
        tried.append(s.train.batch_size)
        if s.train.batch_size > 8:
            raise RuntimeError("MPS backend out of memory (MPS allocated: 24.02 GiB)")
        return {"ok": True}, []

    meta, _ = shrink_on_oom(spec, train, log=lambda _: None)
    assert tried == [32, 16, 8] and meta == {"ok": True, "batch_size": 8}


def test_other_errors_are_not_retried():
    import pytest

    from byosm.spec import Label, TaskSpec
    from byosm.train.gpu import shrink_on_oom

    spec = TaskSpec(name="q", description="q", modality="text", labels=[Label(name="a", definition="")])
    with pytest.raises(RuntimeError, match="shape mismatch"):
        shrink_on_oom(spec, lambda s: (_ for _ in ()).throw(RuntimeError("shape mismatch")), log=lambda _: None)
