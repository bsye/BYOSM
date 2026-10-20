from __future__ import annotations

from typing import Any

import numpy as np
import torch

from ..model import SpecialistModel
from ..runtime import Pool, timed_ms


def latencies(names: list[str], model: SpecialistModel, gmod: Any) -> dict[str, float]:
    pool = Pool.load(names, device="cpu")
    x = next(iter(pool.specialists.values())).mod.example_input()
    ts = timed_ms(lambda: pool.broadcast(x), 30, warmup=5)
    model.to("cpu").eval()
    inputs, _ = gmod.collate([x])

    def shared_forward() -> None:
        feats, mask = model.backbone(**inputs)
        for h in model.heads.values():
            h(feats, mask)

    with torch.no_grad():
        gts = timed_ms(shared_forward, 30, warmup=5)
    return {"specialists_parallel_p50": float(np.median(ts)), "generalist_p50": float(np.median(gts))}
