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


def test_a_second_training_waits_until_the_first_gives_its_share_back(ws, monkeypatch):
    caps = _two_slots(ws, monkeypatch)
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
