"""Load trained specialists and run many of them side by side."""
from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import torch

from . import metrics
from .config import Project
from .modalities import get_modality
from .spec import TaskSpec
from .train import build_model, to_device


class Specialist:
    def __init__(self, run_dir: str | Path, device: str = "cpu"):
        run_dir = Path(run_dir)
        self.spec = TaskSpec.load(run_dir / "spec.yaml")
        self.meta = json.loads((run_dir / "meta.json").read_text())
        self.threshold = self.meta.get("threshold")
        self.mod = get_modality(self.spec)
        self.model = build_model(self.spec, self.mod)
        self.model.load_state_dict(torch.load(run_dir / "model.pt", map_location="cpu"))
        self.device = device
        self.model.to(device).eval()

    @classmethod
    def from_project(cls, name: str, device: str = "cpu") -> "Specialist":
        return cls(Project(name).best_dir, device)

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
                out.append({"frames": [names[i] for i in pred], "probs": pj.round(4).tolist(), "hop_ms": self.spec.frame_hop_ms})
        else:
            pred = metrics.predict(self.spec, p, self.threshold)
            for j in range(len(xs)):
                out.append({"label": names[pred[j]], "probs": {n: round(float(v), 5) for n, v in zip(names, p[j])}})
        return out

    def predict(self, x: Any) -> dict[str, Any]:
        return self.predict_batch([x])[0]

    def latency_ms(self, x: Any | None = None, n: int = 50) -> dict[str, float]:
        x = self.mod.example_input() if x is None else x
        for _ in range(5):
            self.predict(x)
        ts = []
        for _ in range(n):
            t = time.perf_counter()
            self.predict(x)
            if self.device == "mps":
                torch.mps.synchronize()
            ts.append((time.perf_counter() - t) * 1000)
        return {"p50_ms": float(np.median(ts)), "p95_ms": float(np.percentile(ts, 95))}


class Pool:
    """Many tiny System-1 specialists running concurrently on the same stream of inputs."""

    def __init__(self, specialists: list[Specialist], workers: int | None = None):
        self.specialists = {s.name: s for s in specialists}
        self.ex = ThreadPoolExecutor(max_workers=workers or len(specialists))

    @classmethod
    def load(cls, names: list[str], device: str = "cpu") -> "Pool":
        return cls([Specialist.from_project(n, device) for n in names])

    def __call__(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """inputs: {specialist_name: input}. Each specialist gets its own modality's input."""
        futs = {n: self.ex.submit(self.specialists[n].predict, x) for n, x in inputs.items()}
        return {n: f.result() for n, f in futs.items()}

    def broadcast(self, x: Any) -> dict[str, Any]:
        """Same input to every specialist of a matching modality (e.g. one utterance -> intent, toxicity, end-of-turn...)."""
        return self({n: x for n in self.specialists})
