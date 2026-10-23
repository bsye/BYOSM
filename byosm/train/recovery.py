from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ..spec import TaskSpec
from .gpu import LOCK_DIR

OOM_RETRIES = 3
MIN_BATCH = 4
BATCHES, PRECISIONS = "batches", "precisions"
LADDER = ("auto", "bf16", "fp32")
RETRY_IN_PROCESS = ("cuda", "cpu")


class DivergedError(RuntimeError):
    pass


def out_of_memory(e: RuntimeError) -> bool:
    return "out of memory" in str(e).lower()


def known(ws: Path, kind: str) -> dict[str, Any]:
    path = ws / LOCK_DIR / f"{kind}.json"
    return json.loads(path.read_text()) if path.exists() else {}


def remember(ws: Path, kind: str, key: str, value: Any) -> None:
    path = ws / LOCK_DIR / f"{kind}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({**known(ws, kind), key: value}, indent=1))


def model_key(spec: TaskSpec) -> str:
    return f"{spec.modality}|{spec.backbone}"


def _train_with(spec: TaskSpec, **changes: Any) -> TaskSpec:
    return spec.model_copy(update={"train": spec.train.model_copy(update=changes)})


def _child_step(payload: bytes, spec: TaskSpec, device: str, conn: Any) -> None:
    import cloudpickle

    from .gpu import _cap_memory

    try:
        _cap_memory(device, lambda _: None)
        result: tuple[bool, Any] = (True, cloudpickle.loads(payload)(spec))
    except Exception as e:
        result = (False, e)
    try:
        conn.send(result)
    except Exception:
        conn.send((False, RuntimeError(f"{type(result[1]).__name__}: {result[1]}")))


def _in_child(train: Callable[[TaskSpec], tuple[dict, Any]], device: str) -> Callable | None:
    import multiprocessing as mp
    import sys

    import cloudpickle

    try:
        payload = cloudpickle.dumps(train)
    except Exception:
        return None
    ctx = mp.get_context("spawn")

    def attempt(spec: TaskSpec) -> tuple[dict, Any]:
        sys.stdout.flush()
        sys.stderr.flush()
        reader, writer = ctx.Pipe(duplex=False)
        child = ctx.Process(target=_child_step, args=(payload, spec, device, writer))
        child.start()
        writer.close()
        try:
            ok, value = reader.recv()
        except EOFError:
            ok, value = False, None
        child.join()
        if ok:
            return value
        raise value or RuntimeError(f"the training process exited with code {child.exitcode}")

    return attempt


def _remembered(spec: TaskSpec, ws: Path, key: str, log: Callable[[str], None]) -> TaskSpec:
    batch, precision = known(ws, BATCHES).get(key), known(ws, PRECISIONS).get(key)
    if batch and batch < spec.train.batch_size:
        log(f"  batch {batch}: {key} ran out of GPU memory at a larger batch before")
        spec = _train_with(spec, batch_size=batch)
    if precision and spec.train.precision == "auto":
        log(f"  precision {precision}: {key} diverged at a lower precision before")
        spec = _train_with(spec, precision=precision)
    return spec


def _next(spec: TaskSpec, e: RuntimeError, ws: Path, key: str) -> tuple[TaskSpec, str] | None:
    if isinstance(e, DivergedError) and spec.train.precision in LADDER[:-1]:
        precision = LADDER[LADDER.index(spec.train.precision) + 1]
        remember(ws, PRECISIONS, key, precision)
        return _train_with(
            spec, precision=precision
        ), f"diverged at precision {spec.train.precision}; using {precision}"
    size = spec.train.batch_size // 2
    if out_of_memory(e) and size < MIN_BATCH and spec.train.pretrain_steps > 0:
        return _train_with(
            spec, pretrain_steps=0
        ), "out of GPU memory even at the smallest batch; training without pretraining"
    if out_of_memory(e) and size >= MIN_BATCH:
        remember(ws, BATCHES, key, size)
        return _train_with(
            spec, batch_size=size
        ), f"out of GPU memory at batch {spec.train.batch_size}; restarting at batch {size}"
    return None


def recover(spec: TaskSpec, train: Callable[[TaskSpec], tuple[dict, Any]], log: Callable[[str], None], device: str):
    from ..config import find_workspace

    ws, key = find_workspace(), model_key(spec)
    spec = _remembered(spec, ws, key, log)
    attempt = train if device in RETRY_IN_PROCESS else _in_child(train, device)
    for _ in range(OOM_RETRIES + len(LADDER)):
        try:
            meta, rest = (attempt or train)(spec)
            return {**meta, "batch_size": spec.train.batch_size, "precision": spec.train.precision}, rest
        except RuntimeError as e:
            step = _next(spec, e, ws, key)
            if step is None:
                raise
            spec, why = step
            if attempt is None:
                raise RuntimeError(f"{why}; the next run starts there") from e
            log(f"  {why}")
    raise RuntimeError("training failed at every batch size and precision tried")
