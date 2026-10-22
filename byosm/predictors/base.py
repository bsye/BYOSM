from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

import numpy as np

from ..spec import TaskSpec

LATENCY_WARMUP = 3
LATENCY_RUNS = 20
PROBE_RUNS = 1000
PROBE_NOISE_STD = 0.05


class Predictor:
    id: str = ""
    summary: str = ""
    kind: str = "baseline"
    params: int | None = None
    modalities: tuple[str, ...] = ()
    task_types: tuple[str, ...] = ("classify", "frame")
    chunk_ms: float | None = None
    label_space: list[str] | None = None
    hub_repo: str | None = None
    home: str | None = None

    def __init__(self, spec: TaskSpec):
        self.spec = spec

    def probs_batch(self, raws: list[Any]) -> np.ndarray:
        raise NotImplementedError

    def frame_probs(self, raw: Any) -> np.ndarray:
        raise NotImplementedError

    def stream_chunk_ms(self) -> float | None:
        return None

    def latency_ms(self, x: Any) -> float:
        for _ in range(LATENCY_WARMUP):
            self._one(x)
        ts = []
        for _ in range(LATENCY_RUNS):
            t = time.perf_counter()
            self._one(x)
            ts.append((time.perf_counter() - t) * 1000)
        return float(np.median(ts))

    def _one(self, x: Any) -> Any:
        return self.frame_probs(x) if self.spec.task_type == "frame" else self.probs_batch([x])


def frame_hop(spec: TaskSpec, sample_rate: int | None = None) -> int:
    sr = spec.data.audio.sample_rate if sample_rate is None else sample_rate
    return int(sr * spec.frame_hop_ms / 1000)


def probe_noise(n: int) -> np.ndarray:
    return np.random.default_rng(0).normal(0, PROBE_NOISE_STD, (1, n)).astype(np.float32)


BASELINES: dict[str, Callable[[TaskSpec], Predictor]] = {}


def register_baseline(name: str, summary: str) -> Callable[[type[Predictor]], type[Predictor]]:
    def deco(cls: type[Predictor]) -> type[Predictor]:
        cls.id = name
        cls.summary = summary
        BASELINES[name] = cls
        return cls

    return deco


def hub_adapters() -> dict[str, str]:
    return {cls.hub_repo: name for name, cls in BASELINES.items() if getattr(cls, "hub_repo", None)}
