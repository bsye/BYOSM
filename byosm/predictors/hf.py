from __future__ import annotations

from typing import Any

import numpy as np
import torch

from ..spec import TaskSpec
from ..text import match_label
from .base import Predictor

BATCH = 32


def load_model(repo: str, visual: bool) -> tuple[Any, Any]:
    if visual:
        from transformers import AutoImageProcessor, AutoModelForImageClassification

        return AutoImageProcessor.from_pretrained(repo), AutoModelForImageClassification.from_pretrained(repo).eval()
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    return AutoTokenizer.from_pretrained(repo), AutoModelForSequenceClassification.from_pretrained(repo).eval()


class HFClassifier(Predictor):
    def __init__(self, spec: TaskSpec, repo: str):
        super().__init__(spec)
        from transformers import AutoConfig

        self.id = f"hf:{repo}"
        cfg = AutoConfig.from_pretrained(repo)
        self.visual = spec.modality in ("image", "document")
        self.proc, self.model = load_model(repo, self.visual)
        names = spec.label_names
        matched = {int(k): match_label(v, names) for k, v in cfg.id2label.items()}
        self.map = {k: names.index(m) for k, m in matched.items() if m is not None}
        missing = set(range(len(spec.labels))) - set(self.map.values())
        if not self.map:
            theirs = list(cfg.id2label.values())[:8]
            raise ValueError(f"{repo}: none of its labels {theirs} match ours {spec.label_names}")
        self.missing = [spec.label_names[i] for i in sorted(missing)]
        self.params = sum(p.numel() for p in self.model.parameters())

    def _inputs(self, chunk: list[Any]) -> Any:
        if self.visual:
            imgs = [r[0].image if isinstance(r, list) else r for r in chunk]
            return self.proc(images=imgs, return_tensors="pt")
        return self.proc(chunk, padding=True, truncation=True, return_tensors="pt")

    @torch.no_grad()
    def probs_batch(self, raws: list[Any]) -> np.ndarray:
        out = []
        for i in range(0, len(raws), BATCH):
            chunk = raws[i : i + BATCH]
            p = torch.softmax(self.model(**self._inputs(chunk)).logits.float(), -1).numpy()
            q = np.zeros((len(chunk), len(self.spec.labels)), np.float32)
            for src, dst in self.map.items():
                q[:, dst] += p[:, src]
            out.append(q / np.maximum(q.sum(1, keepdims=True), 1e-9))
        return np.concatenate(out)
