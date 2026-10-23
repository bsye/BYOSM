import os
import threading

import pytest

from byosm.train.recovery import BATCHES, DivergedError, known, model_key, recover


def _spec():
    from byosm.spec import Label, TaskSpec

    return TaskSpec(name="q", description="q", modality="text", labels=[Label(name="a", definition="")])


def _logged(ws):
    path = ws / "attempts.txt"
    return [line.split() for line in path.read_text().splitlines()] if path.exists() else []


def _fits_at(ws, fits_at):
    def train(s):
        with (ws / "attempts.txt").open("a") as f:
            f.write(f"{os.getpid()} {s.train.batch_size} {s.train.precision}\n")
        if s.train.precision != "fp32" and fits_at == "fp32":
            raise DivergedError("NaN")
        if fits_at != "fp32" and s.train.batch_size > fits_at:
            raise RuntimeError("MPS backend out of memory (MPS allocated: 24.02 GiB)")
        return {"pid": os.getpid()}, []

    return train


def test_on_apple_gpus_each_attempt_trains_in_its_own_process_and_the_caller_keeps_going(ws):
    meta, _ = recover(_spec(), _fits_at(ws, 16), log=lambda _: None, device="mps")
    attempts = _logged(ws)
    assert [a[1] for a in attempts] == ["32", "16"] and meta["batch_size"] == 16
    pids = {a[0] for a in attempts}
    assert len(pids) == 2 and str(os.getpid()) not in pids and meta["pid"] != os.getpid()


def test_on_apple_gpus_divergence_climbs_the_precision_ladder_in_fresh_processes(ws):
    meta, _ = recover(_spec(), _fits_at(ws, "fp32"), log=lambda _: None, device="mps")
    assert [a[2] for a in _logged(ws)] == ["auto", "bf16", "fp32"] and meta["precision"] == "fp32"


def test_a_training_process_that_dies_fails_the_step_not_the_caller(ws):
    def train(s):
        os._exit(3)

    with pytest.raises(RuntimeError, match="exited with code 3"):
        recover(_spec(), train, log=lambda _: None, device="mps")


def test_errors_from_the_training_process_keep_their_type(ws):
    def train(s):
        raise ValueError("shape mismatch")

    with pytest.raises(ValueError, match="shape mismatch"):
        recover(_spec(), train, log=lambda _: None, device="mps")


def test_a_step_that_cannot_run_in_its_own_process_stops_with_the_halved_batch_remembered(ws):
    lock = threading.Lock()

    def train(s):
        with lock:
            raise RuntimeError("MPS backend out of memory")

    with pytest.raises(RuntimeError, match="next run starts there"):
        recover(_spec(), train, log=lambda _: None, device="mps")
    assert known(ws, BATCHES)[model_key(_spec())] == 16


def test_on_the_cpu_running_out_of_memory_retries_in_the_same_process(ws):
    tried = []

    def train(s):
        tried.append(s.train.batch_size)
        if s.train.batch_size > 16:
            raise RuntimeError("out of memory")
        return {"pid": os.getpid()}, []

    meta, _ = recover(_spec(), train, log=lambda _: None, device="cpu")
    assert tried == [32, 16] and meta["pid"] == os.getpid()
