"""One interface for everything that gets scored: our experiments and external baselines.

A Predictor returns probabilities over the spec's labels:
  classify -> probs_batch(raws) -> [N, C]
  frame    -> frame_probs(raw)  -> [T, C] at the spec's frame hop

Add a baseline with @register_baseline("name"). It is then usable from spec.yaml (`baselines: [name]`) and from the CLI.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

from .spec import TaskSpec


class Predictor:
    id: str = ""
    kind: str = "baseline"
    params: int | None = None
    modalities: tuple[str, ...] = ()
    task_types: tuple[str, ...] = ("classify", "frame")

    def __init__(self, spec: TaskSpec):
        self.spec = spec

    def probs_batch(self, raws: list[Any]) -> np.ndarray:
        raise NotImplementedError

    def frame_probs(self, raw: Any) -> np.ndarray:
        raise NotImplementedError

    def stream_chunk_ms(self) -> float | None:
        """CPU time to process one streaming chunk (frame tasks), if the predictor streams."""
        return None

    chunk_ms: float | None = None

    def latency_ms(self, x: Any) -> float:
        for _ in range(3):
            self._one(x)
        ts = []
        for _ in range(20):
            t = time.perf_counter()
            self._one(x)
            ts.append((time.perf_counter() - t) * 1000)
        return float(np.median(ts))

    def _one(self, x: Any) -> Any:
        return self.frame_probs(x) if self.spec.task_type == "frame" else self.probs_batch([x])


class ExperimentPredictor(Predictor):
    kind = "experiment"

    def __init__(self, spec: TaskSpec, run_dir: Path, name: str):
        super().__init__(spec)
        from .runtime import Specialist

        self.sp = Specialist(run_dir, "cpu")
        self.id = name
        self.params = int(self.sp.meta["params"])
        self.bb = self.sp.model.backbone
        self.head = next(iter(self.sp.model.heads.values()))
        self.chunk_ms = 30.0 if hasattr(self.bb, "stream") else None

    @torch.no_grad()
    def probs_batch(self, raws: list[Any]) -> np.ndarray:
        out = []
        for i in range(0, len(raws), 64):
            inputs, _ = self.sp.mod.collate(raws[i : i + 64])
            logits, _ = self.sp.model(inputs)
            out.append(torch.softmax(logits.float(), -1).numpy())
        return np.concatenate(out)

    @torch.no_grad()
    def frame_probs(self, raw: Any) -> np.ndarray:
        if hasattr(self.bb, "stream"):  # continuous streaming, like deployment
            hop = self.bb.hop
            n = len(raw) // hop * hop
            state, out = None, []
            for i in range(0, n, 160_000):
                f, state = self.bb.stream(torch.from_numpy(np.ascontiguousarray(raw[i : min(n, i + 160_000)]))[None], state)
                out.append(torch.softmax(self.head.net(f), -1)[0].numpy())
            return np.concatenate(out) if out else np.zeros((0, len(self.spec.labels)), np.float32)
        # Non-streaming backbones: independent windows of the training clip length.
        sr = self.spec.data.audio.sample_rate
        w = int(sr * self.spec.data.audio.clip_seconds)
        hop = int(sr * self.spec.frame_hop_ms / 1000)
        out = []
        for i in range(0, len(raw), w):
            seg = raw[i : i + w]
            if len(seg) < hop:
                break
            inputs, _ = self.sp.mod.collate([seg])
            logits, mask = self.sp.model(inputs)
            p = torch.softmax(logits.float(), -1)[0][mask[0]].numpy()
            idx = np.minimum((np.arange(len(seg) // hop) * len(p)) // max(1, len(seg) // hop), len(p) - 1)
            out.append(p[idx])
        return np.concatenate(out)

    @torch.no_grad()
    def stream_chunk_ms(self) -> float | None:
        if not hasattr(self.bb, "stream"):
            return None
        x = torch.from_numpy(np.random.default_rng(0).normal(0, 0.05, (1, 480)).astype(np.float32))
        threads = torch.get_num_threads()
        torch.set_num_threads(1)  # single core, like Silero in LiveKit
        try:
            _, state = self.bb.stream(x)
            t = time.perf_counter()
            for _ in range(1000):
                f, state = self.bb.stream(x, state)
                self.head.net(f)
            return time.perf_counter() - t
        finally:
            torch.set_num_threads(threads)


BASELINES: dict[str, Callable[[TaskSpec], Predictor]] = {}


def register_baseline(name: str) -> Callable[[type[Predictor]], type[Predictor]]:
    def deco(cls: type[Predictor]) -> type[Predictor]:
        cls.id = name
        BASELINES[name] = cls
        return cls

    return deco


def get_baseline(name: str, spec: TaskSpec) -> Predictor:
    if name not in BASELINES:
        raise KeyError(f"unknown baseline {name!r}; available: {sorted(BASELINES)}")
    cls = BASELINES[name]
    if cls.modalities and spec.modality not in cls.modalities or spec.task_type not in cls.task_types:
        raise ValueError(f"baseline {name!r} supports {cls.modalities} / {cls.task_types}, not {spec.modality} / {spec.task_type}")
    return cls(spec)


@register_baseline("silero")
class SileroVAD(Predictor):
    """Silero VAD, the exact ONNX bundled with livekit-plugins-silero (LiveKit's VAD)."""

    modalities, task_types = ("audio",), ("frame",)
    window = 512
    chunk_ms = 32.0

    def __init__(self, spec: TaskSpec):
        super().__init__(spec)
        try:
            from livekit.plugins.silero.onnx_model import OnnxModel, new_inference_session
        except ImportError as e:
            raise ImportError("pip install livekit-plugins-silero onnxruntime") from e
        self.m = OnnxModel(onnx_session=new_inference_session(force_cpu=True), sample_rate=16000)
        self.params = 545_601  # both 8k and 16k branches of the bundled ONNX

    def frame_probs(self, raw: Any) -> np.ndarray:
        self.m.reset()
        n = len(raw) // self.window
        wp = np.array([self.m(np.ascontiguousarray(raw[i * self.window : (i + 1) * self.window])[None]) for i in range(n)], np.float32)
        hop = int(16000 * self.spec.frame_hop_ms / 1000)
        # frame -> window containing the frame start (up to 32ms look-ahead: favours Silero)
        idx = np.minimum(np.arange(len(raw) // hop) * hop // self.window, max(0, len(wp) - 1))
        p = wp[idx] if len(wp) else np.zeros(len(idx), np.float32)
        out = np.zeros((len(p), len(self.spec.labels)), np.float32)
        out[:, self.spec.pos_index] = p
        out[:, [i for i in range(len(self.spec.labels)) if i != self.spec.pos_index][0]] = 1 - p
        return out

    def stream_chunk_ms(self) -> float:
        self.m.reset()
        x = np.random.default_rng(0).normal(0, 0.05, (1, self.window)).astype(np.float32)
        t = time.perf_counter()
        for _ in range(1000):
            self.m(x)
        return time.perf_counter() - t


@register_baseline("energy")
class EnergyVAD(Predictor):
    """Floor baseline: frame log-energy squashed to [0,1]. If a model can't beat this, something is wrong."""

    modalities, task_types = ("audio",), ("frame",)
    params = 0

    def frame_probs(self, raw: Any) -> np.ndarray:
        hop = int(16000 * self.spec.frame_hop_ms / 1000)
        n = len(raw) // hop
        e = 10 * np.log10(np.mean(np.asarray(raw[: n * hop]).reshape(n, hop) ** 2, 1) + 1e-10)
        p = 1 / (1 + np.exp(-(e + 45) / 4))
        out = np.zeros((n, len(self.spec.labels)), np.float32)
        out[:, self.spec.pos_index] = p
        out[:, [i for i in range(len(self.spec.labels)) if i != self.spec.pos_index][0]] = 1 - p
        return out


@register_baseline("llm")
class ZeroShotLLM(Predictor):
    """The LLM itself, zero-shot, given the label definitions (role `baseline` in byojl.yaml). The 'big System 2' bar."""

    modalities, task_types = ("text",), ("classify",)

    def probs_batch(self, raws: list[Any]) -> np.ndarray:
        from .data.synth_text import JUDGE_SYSTEM, _context
        from .llm import get_llm, pmap

        llm = get_llm("baseline")
        self.id = f"llm:{llm.name}"
        names = self.spec.label_names
        schema = {"type": "object", "properties": {"labels": {"type": "array", "items": {"type": "string", "enum": [*names, "__drop__"]}}}}
        chunks = [raws[i : i + 40] for i in range(0, len(raws), 40)]

        def run(chunk: list[str]) -> list[str]:
            listing = "\n".join(f"{i + 1}. {r}" for i, r in enumerate(chunk))
            labs = llm.json(JUDGE_SYSTEM, f"{_context(self.spec)}\n\nInputs:\n{listing}\n\nReturn {len(chunk)} labels.", schema).get("labels", [])
            return (labs + [names[0]] * len(chunk))[: len(chunk)]

        labs = [l for part in pmap(run, chunks, 8) for l in part]
        out = np.full((len(raws), len(names)), 0.0, np.float32)
        for i, l in enumerate(labs):
            out[i, names.index(l) if l in names else 0] = 1.0
        return out
