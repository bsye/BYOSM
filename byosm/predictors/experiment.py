from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ..rt import first_subtokens
from ..spec import TaskSpec
from .base import PROBE_RUNS, Predictor, frame_hop, probe_noise

BATCH = 64
STREAM_SPAN = 160_000
STREAM_CHUNK_MS = 30.0
PROBE_CHUNK = 480


class ExperimentPredictor(Predictor):
    kind = "experiment"

    def __init__(self, spec: TaskSpec, meta: dict, run_dir: Path, name: str, device: str = "cpu"):
        super().__init__(spec)
        from ..runtime import Specialist

        self.sp = Specialist(spec, meta, run_dir, device)
        self.device = device
        self.id = name
        self.params = int(self.sp.meta["params"])
        self.bb = self.sp.model.backbone
        self.head = next(iter(self.sp.model.heads.values()))
        self.chunk_ms = STREAM_CHUNK_MS if hasattr(self.bb, "stream") else None

    @torch.no_grad()
    def probs_batch(self, raws: list[Any]) -> np.ndarray:
        out = []
        for i in range(0, len(raws), BATCH):
            inputs, _ = self.sp.mod.collate(raws[i : i + BATCH])
            logits, _ = self.sp.model({k: v.to(self.device) for k, v in inputs.items()})
            out.append(torch.softmax(logits.float(), -1).cpu().numpy())
        return np.concatenate(out)

    @torch.no_grad()
    def frame_probs(self, raw: Any) -> np.ndarray:
        if self.spec.modality == "text":
            return self._word_frames(raw)
        if hasattr(self.bb, "stream"):
            return self._streamed_frames(raw)
        return self._windowed_frames(raw)

    def _word_frames(self, raw: str) -> np.ndarray:
        enc = self.sp.mod.encode_words([raw])
        logits, _ = self.sp.model({k: v.to(self.device) for k, v in enc.items()})
        return torch.softmax(logits.float(), -1)[0].cpu().numpy()[first_subtokens(enc.word_ids(0))]

    def _streamed_frames(self, raw: Any) -> np.ndarray:
        n = len(raw) // self.bb.hop * self.bb.hop
        state, out = None, []
        for i in range(0, n, STREAM_SPAN):
            span = torch.from_numpy(np.ascontiguousarray(raw[i : min(n, i + STREAM_SPAN)]))[None]
            f, state = self.bb.stream(span, state)
            out.append(torch.softmax(self.head.net(f), -1)[0].numpy())
        return np.concatenate(out) if out else np.zeros((0, len(self.spec.labels)), np.float32)

    def _windowed_frames(self, raw: Any) -> np.ndarray:
        sr = self.spec.data.audio.sample_rate
        w = int(sr * self.spec.data.audio.clip_seconds)
        hop = frame_hop(self.spec)
        out = []
        for i in range(0, len(raw), w):
            seg = raw[i : i + w]
            if len(seg) < hop:
                break
            inputs, _ = self.sp.mod.collate([seg])
            logits, mask = self.sp.model(inputs)
            p = torch.softmax(logits.float(), -1)[0][mask[0]].numpy()
            frames = len(seg) // hop
            idx = np.minimum((np.arange(frames) * len(p)) // max(1, frames), len(p) - 1)
            out.append(p[idx])
        return np.concatenate(out)

    @torch.no_grad()
    def stream_chunk_ms(self) -> float | None:
        if not hasattr(self.bb, "stream"):
            return None
        x = torch.from_numpy(probe_noise(PROBE_CHUNK))
        threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            _, state = self.bb.stream(x)
            t = time.perf_counter()
            for _ in range(PROBE_RUNS):
                f, state = self.bb.stream(x, state)
                self.head.net(f)
            return time.perf_counter() - t
        finally:
            torch.set_num_threads(threads)
