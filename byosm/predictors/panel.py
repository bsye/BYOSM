from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ..spec import TaskSpec
from .base import Predictor, register_baseline


@register_baseline(
    "panel", "The teacher panel itself: each teacher's answer, weighted by its accuracy on the labelled rows."
)
class Panel(Predictor):
    task_types = ("classify",)
    wants_rows = True

    def __init__(self, spec: TaskSpec):
        super().__init__(spec)
        from ..config import Project
        from ..data.panel import load_panel

        self.stats = load_panel(Project(spec.name).data_dir_for(spec))
        if not self.stats:
            raise ValueError(f"{spec.name} has no teacher panel: set data.teachers and rebuild its data")
        self.name = "panel:" + "+".join(self.stats["teachers"])

    def probs_rows(self, rows: list[dict], mod: Any) -> np.ndarray:
        from ..data.panel import combine, teacher_probs

        root = getattr(mod, "data_dir", None)
        teachers = self.stats["teachers"]
        probs = [teacher_probs(self.spec, t, rows, Path(root) if root else None, print) for t in teachers]
        return combine(probs, list(teachers.values()), np.asarray(self.stats["prior"]))

    def latency_ms(self, x: Any) -> float:
        return float("nan")
