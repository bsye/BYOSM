from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

from ..events import emit
from .combine import combine
from .models import Boosted, Group, rivals
from .patch import PatchModel, PretrainedPatch, pretrained
from .scoring import Progress, score_model, scored
from .specialist import is_rival

FITTING_EVENT = "fitting"


def candidates(effort: int) -> list[Any]:
    base = [PretrainedPatch(effort)] if pretrained() else []
    return [*rivals(), Boosted(effort), PatchModel(effort), *base]


def fit_domain(
    configs: list[str], effort: int, log: Callable[[str], None], progress: Progress
) -> tuple[dict[str, Any], dict[str, Any]]:
    s = scored(configs, log)
    groups = [Group(p.histories, p.season, p.horizon) for p in s.prepared]
    models = candidates(effort)
    for i, model in enumerate(models):
        log(f"  training {model.name}, recipe {i + 1} of {len(models)}")
        emit(FITTING_EVENT, recipe=model.name, index=i + 1, of=len(models))
        t0 = time.perf_counter()
        model.fit(groups)
        took = time.perf_counter() - t0
        log(f"  {model.name}: one model for {len(s.prepared)} datasets, trained in {took:.1f} s")
        score_model(model, s, took, log)
        progress(s.per, [m.name for m in models[i + 1 :]])
    return s.per, combine(s, [m for m in models if not is_rival(m.name)])
