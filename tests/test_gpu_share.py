import threading

import yaml

import byosm.train.gpu as gpu


def _two_slots(ws, monkeypatch):
    cfg = yaml.safe_load((ws / "byosm.yaml").read_text())
    (ws / "byosm.yaml").write_text(yaml.safe_dump({**cfg, "gpu_jobs": 2}))
    monkeypatch.setattr(gpu, "free_gb", lambda: 100.0)
    monkeypatch.setattr(gpu, "WAIT_S", 0.01)
    caps = []
    monkeypatch.setattr(gpu, "_cap_memory", lambda device, log, share=None: caps.append(share))
    return caps


def test_a_training_alone_gets_most_of_the_gpu_and_gives_it_back(ws, monkeypatch):
    caps = _two_slots(ws, monkeypatch)
    with gpu.gpu_slot("mps", log=lambda _: None):
        assert round(gpu.held_share(ws), 2) == gpu.MPS_SHARE
    assert caps == [gpu.MPS_SHARE] and gpu.held_share(ws) == 0.0


def test_a_second_training_waits_while_memory_is_really_short(ws, monkeypatch):
    caps = _two_slots(ws, monkeypatch)
    monkeypatch.setattr(gpu, "free_gb", lambda: 4.0)
    started, release, done = threading.Event(), threading.Event(), threading.Event()

    def first():
        with gpu.gpu_slot("mps", log=lambda _: None):
            started.set()
            release.wait(5)

    def second():
        with gpu.gpu_slot("mps", log=lambda _: None):
            done.set()

    a = threading.Thread(target=first)
    a.start()
    started.wait(5)
    b = threading.Thread(target=second)
    b.start()
    assert not done.wait(0.3)
    release.set()
    assert done.wait(5)
    a.join()
    b.join()
    assert caps == [gpu.MPS_SHARE, gpu.MPS_SHARE]


def test_out_of_memory_at_the_smallest_batch_drops_pretraining_before_giving_up(ws):
    from byosm.spec import Label, TaskSpec
    from byosm.train.recovery import MIN_BATCH, recover

    spec = TaskSpec(name="q", description="q", modality="text", labels=[Label(name="a", definition="")])
    spec.train.batch_size, spec.train.pretrain_steps = MIN_BATCH, 200
    tried = []

    def train(s):
        tried.append(s.train.pretrain_steps)
        if s.train.pretrain_steps:
            raise RuntimeError("MPS backend out of memory")
        return {"ok": True}, None

    recover(spec, train, log=lambda _: None, device="cuda")
    assert tried == [200, 0]


def test_pretraining_reads_shorter_text_than_fine_tuning():
    from byosm.modalities import get_modality
    from byosm.spec import Label, TaskSpec
    from byosm.train.pretraining import PRETRAIN_MAX_LEN, _short_views

    spec = TaskSpec(name="q", description="q", modality="text", labels=[Label(name="a", definition="")])
    spec.train.max_len = 576
    mod = get_modality(spec)
    assert _short_views(spec, mod).spec.train.max_len == PRETRAIN_MAX_LEN and spec.train.max_len == 576


def test_a_model_that_outputs_nan_raises_the_error_recovery_climbs_the_precision_ladder_for(monkeypatch):
    import numpy as np
    import pytest

    import byosm.train.evaluate as evaluate
    from byosm.spec import Label, TaskSpec
    from byosm.train.recovery import DivergedError

    spec = TaskSpec(
        name="q",
        description="q",
        modality="text",
        labels=[Label(name="a", definition=""), Label(name="b", definition="")],
    )
    monkeypatch.setattr(evaluate, "targets_of", lambda spec, mod, rows: [0])
    monkeypatch.setattr(evaluate, "infer", lambda *a, **k: {"y": np.array([0]), "probs": np.array([[np.nan, np.nan]])})
    with pytest.raises(DivergedError):
        evaluate.evaluate_rows(None, None, spec, [{}], ["x"], "mps", None)


def test_each_recovery_is_reported_as_an_event(ws, monkeypatch, capsys):
    from byosm.events import EVENTS_ENV, parse
    from byosm.spec import Label, TaskSpec
    from byosm.train.recovery import RECOVERY, recover

    monkeypatch.setenv(EVENTS_ENV, "1")
    spec = TaskSpec(name="q", description="q", modality="text", labels=[Label(name="a", definition="")])
    tries = iter([RuntimeError("out of memory"), None])

    def train(s):
        if (e := next(tries)) is not None:
            raise e
        return {"ok": True}, None

    recover(spec, train, log=lambda _: None, device="cuda")
    events = [e for e in map(parse, capsys.readouterr().out.splitlines()) if e and e["kind"] == RECOVERY]
    assert (
        len(events) == 1
        and "out of GPU memory" in events[0]["why"]
        and events[0]["batch"] == spec.train.batch_size // 2
    )


def test_trainings_starting_together_when_memory_is_short_cannot_both_reserve_most_of_the_gpu(ws, monkeypatch):
    _two_slots(ws, monkeypatch)
    monkeypatch.setattr(gpu, "free_gb", lambda: 4.0)
    reserved = []
    monkeypatch.setattr(gpu, "_cap_memory", lambda device, log, share=None: reserved.append(gpu.held_share(ws)))
    release, ready = threading.Event(), threading.Barrier(2)

    def train():
        ready.wait(5)
        with gpu.gpu_slot("mps", log=lambda _: None):
            release.wait(0.5)

    threads = [threading.Thread(target=train) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(reserved) == 2 and max(reserved) <= gpu.MPS_SHARE + 1e-6


def test_the_gpu_queue_shows_which_job_holds_the_gpu_and_its_share(ws, monkeypatch):
    _two_slots(ws, monkeypatch)
    monkeypatch.setenv(gpu.JOB_ENV, "job-1")
    with gpu.gpu_slot("mps", log=lambda _: None):
        [held] = gpu.holders(ws)
        assert held["job"] == "job-1" and round(held["share"], 2) == gpu.MPS_SHARE and held["since"] > 0
    assert gpu.holders(ws) == []


def test_a_small_training_runs_on_the_cpu_instead_of_waiting_for_a_busy_gpu(ws, monkeypatch):
    _two_slots(ws, monkeypatch)
    monkeypatch.setattr(gpu, "free_gb", lambda: 4.0)
    with (
        gpu.gpu_slot("mps", log=lambda _: None) as first,
        gpu.gpu_slot("mps", log=lambda _: None, fallback="cpu") as second,
    ):
        assert (first, second) == ("mps", "cpu")
    with gpu.gpu_slot("mps", log=lambda _: None, fallback="cpu") as alone:
        assert alone == "mps"


def test_a_second_training_starts_when_memory_is_free_even_if_the_first_reserved_most(ws, monkeypatch):
    _two_slots(ws, monkeypatch)
    monkeypatch.setattr(gpu, "free_gb", lambda: 32.0)
    monkeypatch.setattr("psutil.virtual_memory", lambda: type("M", (), {"total": 64 * 1024**3})())
    with gpu.gpu_slot("mps", log=lambda _: None), gpu.gpu_slot("mps", log=lambda _: None, fallback="cpu") as second:
        assert second == "mps"


def test_the_gpu_goes_to_whoever_has_waited_longest(tmp_path):
    import os

    from byosm.train.gpu_queue import QUEUE, TICKET, first, join

    mine = join(tmp_path)
    older = tmp_path / QUEUE / f"{os.getppid()}{TICKET}"
    older.write_text("1.0")
    assert not first(mine)
    older.unlink()
    (tmp_path / QUEUE / f"999999{TICKET}").write_text("0.5")
    assert first(mine) and not (tmp_path / QUEUE / f"999999{TICKET}").exists()
