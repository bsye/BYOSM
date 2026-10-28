from __future__ import annotations

from typing import Any

import numpy as np

from .base import Predictor, register_baseline

ALWAYS_POSITIVE = "always-positive"


@register_baseline(
    ALWAYS_POSITIVE,
    "Floor baseline: the positive label for every input. Its score follows from the labels alone, so a public suite "
    "checks against it that our conversion and scoring reproduce the benchmark's own numbers.",
)
class AlwaysPositive(Predictor):
    task_types = ("classify",)
    params = 0

    def probs_batch(self, raws: list[Any]) -> np.ndarray:
        out = np.zeros((len(raws), len(self.spec.labels)), np.float32)
        out[:, self.spec.pos_index] = 1.0
        return out
