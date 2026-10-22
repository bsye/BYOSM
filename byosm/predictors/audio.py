from __future__ import annotations

import time
from typing import Any

import numpy as np

from ..spec import TaskSpec
from .base import PROBE_RUNS, Predictor, frame_hop, probe_noise, register_baseline

SAMPLE_RATE = 16000
SILERO_PARAMS = 545_601
ENERGY_CENTER_DB = -45
ENERGY_SCALE_DB = 4


def binary_frames(spec: TaskSpec, p: np.ndarray) -> np.ndarray:
    out = np.zeros((len(p), len(spec.labels)), np.float32)
    out[:, spec.pos_index] = p
    out[:, [i for i in range(len(spec.labels)) if i != spec.pos_index][0]] = 1 - p
    return out


def silero_model() -> Any:
    try:
        from livekit.plugins.silero.onnx_model import OnnxModel, new_inference_session
    except ImportError as e:
        raise ImportError("pip install livekit-plugins-silero onnxruntime") from e
    return OnnxModel(onnx_session=new_inference_session(force_cpu=True), sample_rate=SAMPLE_RATE)


@register_baseline("silero", "Silero VAD, the exact ONNX bundled with livekit-plugins-silero (LiveKit's VAD).")
class SileroVAD(Predictor):
    modalities, task_types = ("audio",), ("frame",)
    window = 512
    chunk_ms = 32.0

    def __init__(self, spec: TaskSpec):
        super().__init__(spec)
        self.m = silero_model()
        self.params = SILERO_PARAMS

    def frame_probs(self, raw: Any) -> np.ndarray:
        self.m.reset()
        n = len(raw) // self.window
        windows = (np.ascontiguousarray(raw[i * self.window : (i + 1) * self.window])[None] for i in range(n))
        wp = np.array([self.m(w) for w in windows], np.float32)
        hop = frame_hop(self.spec, SAMPLE_RATE)
        idx = np.minimum(np.arange(len(raw) // hop) * hop // self.window, max(0, len(wp) - 1))
        p = wp[idx] if len(wp) else np.zeros(len(idx), np.float32)
        return binary_frames(self.spec, p)

    def stream_chunk_ms(self) -> float:
        self.m.reset()
        x = probe_noise(self.window)
        t = time.perf_counter()
        for _ in range(PROBE_RUNS):
            self.m(x)
        return time.perf_counter() - t


@register_baseline(
    "energy",
    "Floor baseline: frame log-energy squashed to [0,1]. If a model can't beat this, something is wrong.",
)
class EnergyVAD(Predictor):
    modalities, task_types = ("audio",), ("frame",)
    params = 0

    def frame_probs(self, raw: Any) -> np.ndarray:
        hop = frame_hop(self.spec, SAMPLE_RATE)
        n = len(raw) // hop
        e = 10 * np.log10(np.mean(np.asarray(raw[: n * hop]).reshape(n, hop) ** 2, 1) + 1e-10)
        p = 1 / (1 + np.exp(-(e - ENERGY_CENTER_DB) / ENERGY_SCALE_DB))
        return binary_frames(self.spec, p)
