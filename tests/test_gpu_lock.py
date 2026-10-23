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


def _spec():
    from byosm.spec import Label, TaskSpec

    return TaskSpec(name="q", description="q", modality="text", labels=[Label(name="a", definition="")])


def _train(tried, fits_at):
    def train(s):
        tried.append(s.train.batch_size)
        if s.train.batch_size > fits_at:
            raise RuntimeError("MPS backend out of memory (MPS allocated: 24.02 GiB)")
        return {"ok": True}, []

    return train


def test_on_cuda_running_out_of_memory_halves_the_batch_until_it_fits(ws):
    from byosm.train.recovery import recover as shrink_on_oom

    tried = []
    meta, _ = shrink_on_oom(_spec(), _train(tried, 8), log=lambda _: None, device="cuda")
    assert tried == [32, 16, 8] and meta == {"ok": True, "batch_size": 8, "precision": "auto"}


def test_on_apple_gpus_the_job_restarts_itself_at_half_the_batch(ws, monkeypatch):
    import pytest

    from byosm.train.recovery import recover as shrink_on_oom

    def restart(log, why):
        raise SystemExit(why)

    import byosm.train.recovery as recovery

    monkeypatch.setattr(recovery, "restart", restart)
    tried = []
    with pytest.raises(SystemExit, match="restarting at batch 16"):
        shrink_on_oom(_spec(), _train(tried, 16), log=lambda _: None, device="mps")
    meta, _ = shrink_on_oom(_spec(), _train(tried, 16), log=lambda _: None, device="mps")
    assert tried == [32, 16] and meta["batch_size"] == 16


def test_other_errors_are_not_retried(ws):
    import pytest

    from byosm.spec import Label, TaskSpec
    from byosm.train.recovery import recover as shrink_on_oom

    spec = TaskSpec(name="q", description="q", modality="text", labels=[Label(name="a", definition="")])
    with pytest.raises(RuntimeError, match="shape mismatch"):
        shrink_on_oom(
            spec, lambda s: (_ for _ in ()).throw(RuntimeError("shape mismatch")), log=lambda _: None, device="cuda"
        )


def test_a_training_that_diverges_climbs_the_precision_ladder_and_is_remembered(ws):
    from byosm.train.recovery import DivergedError, recover

    tried = []

    def train(spec):
        tried.append(spec.train.precision)
        if spec.train.precision != "fp32":
            raise DivergedError("NaN")
        return {"ok": True}, None

    meta, _ = recover(_spec(), train, log=lambda _: None, device="cuda")
    assert tried == ["auto", "bf16", "fp32"] and meta["precision"] == "fp32"
    tried.clear()
    recover(_spec(), train, log=lambda _: None, device="cuda")
    assert tried == ["fp32"]
