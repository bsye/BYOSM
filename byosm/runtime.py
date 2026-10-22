from __future__ import annotations

import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import torch

from . import metrics
from .modalities import get_modality
from .project import Project
from .spec import TaskSpec
from .train import build_model, to_device


def timed_ms(fn: Callable[[], Any], n: int, warmup: int = 0, sync: Callable[[], Any] | None = None) -> list[float]:
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(n):
        t = time.perf_counter()
        fn()
        if sync is not None:
            sync()
        ts.append((time.perf_counter() - t) * 1000)
    return ts


def _weights(spec: TaskSpec, run_dir: Path) -> Path:
    path = run_dir / "model.pt"
    if path.exists():
        return path
    from .artifacts import weights

    return weights(Project(spec.name), run_dir.name)


class Specialist:
    def __init__(self, spec: TaskSpec, meta: dict, run_dir: Path, device: str = "cpu"):
        self.spec, self.meta = spec, meta
        self.threshold = self.meta.get("threshold")
        self.mod = get_modality(self.spec)
        self.model = build_model(self.spec, self.mod)
        self.model.load_state_dict(torch.load(_weights(spec, run_dir), map_location="cpu"))
        self.device = device
        self.model.to(device).eval()

    @classmethod
    def from_project(cls, name: str, device: str = "cpu") -> Specialist:
        from .evals.selection import calibrated_threshold

        p = Project(name)
        sp = cls(p.exp_spec(p.promoted), p.meta(p.promoted), p.best_dir, device)
        sp.threshold = calibrated_threshold(p, p.promoted)
        return sp

    @property
    def name(self) -> str:
        return self.spec.name

    @torch.no_grad()
    def predict_batch(self, xs: list[Any]) -> list[dict[str, Any]]:
        inputs, _ = self.mod.collate(xs)
        logits, mask = self.model(to_device(inputs, self.device))
        p = torch.softmax(logits.float(), -1).cpu().numpy()
        names = self.spec.label_names
        out = []
        if self.spec.task_type == "frame":
            mask = mask.cpu().numpy()
            for j in range(len(xs)):
                pj = p[j][mask[j]]
                pred = metrics.predict(self.spec, pj, self.threshold)
                out.append(
                    {
                        "frames": [names[i] for i in pred],
                        "probs": pj.round(4).tolist(),
                        "hop_ms": self.spec.frame_hop_ms,
                    }
                )
        else:
            pred = metrics.predict(self.spec, p, self.threshold)
            for j in range(len(xs)):
                out.append(
                    {
                        "label": names[pred[j]],
                        "probs": {n: round(float(v), 5) for n, v in zip(names, p[j], strict=False)},
                    }
                )
        return out

    def predict(self, x: Any) -> dict[str, Any]:
        return self.predict_batch([x])[0]

    def latency_ms(self, x: Any | None = None, n: int = 50) -> dict[str, float]:
        x = self.mod.example_input() if x is None else x
        sync = torch.mps.synchronize if self.device == "mps" else None
        ts = timed_ms(lambda: self.predict(x), n, warmup=5, sync=sync)
        return {"p50_ms": float(np.median(ts)), "p95_ms": float(np.percentile(ts, 95))}


class Pool:
    def __init__(self, specialists: list[Specialist], workers: int | None = None):
        self.specialists = {s.name: s for s in specialists}
        self.ex = ThreadPoolExecutor(max_workers=workers or len(specialists))

    @classmethod
    def load(cls, names: list[str], device: str = "cpu") -> Pool:
        return cls([Specialist.from_project(n, device) for n in names])

    def __call__(self, inputs: dict[str, Any]) -> dict[str, Any]:
        futs = {n: self.ex.submit(self.specialists[n].predict, x) for n, x in inputs.items()}
        return {n: f.result() for n, f in futs.items()}

    def broadcast(self, x: Any) -> dict[str, Any]:
        return self({n: x for n in self.specialists})
