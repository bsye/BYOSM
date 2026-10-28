from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

from ..events import emit
from .combine import combine
from .models import Boosted, Group, SeasonalNaive, rivals
from .patch import PatchModel, PretrainedPatch, pretrained
from .prediction import PredictCfg
from .scoring import Progress, score_model, scored
from .specialist import is_rival

FITTING_EVENT = "fitting"


def candidates(effort: int) -> list[Any]:
    base = [PretrainedPatch(effort)] if pretrained() else []
    return [*rivals(), Boosted(effort), PatchModel(effort), *base]


def fit_domain(
    configs: list[str], effort: int, log: Callable[[str], None], progress: Progress, cfg: PredictCfg
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
    return s.per, combine(s, [m for m in models if not is_rival(m.name)], cfg, log)


def _train_s(parent: dict[str, dict[str, Any]], name: str) -> float:
    return next((r[name]["train_s"] for r in parent.values() if "train_s" in r.get(name, {})), 0.0)


def rescore(
    configs: list[str],
    models: list[Any],
    parent: dict[str, dict[str, Any]],
    log: Callable[[str], None],
    progress: Progress,
    cfg: PredictCfg,
) -> tuple[dict[str, Any], dict[str, Any]]:
    s = scored(configs, log)
    every = [SeasonalNaive(), *models]
    for i, model in enumerate(every):
        score_model(model, s, _train_s(parent, model.name), log)
        progress(s.per, [m.name for m in every[i + 1 :]])
    for config, row in s.per.items():
        kept = parent.get(config, {}).items()
        row |= {m: r for m, r in kept if is_rival(m) and m not in row and not r.get("published")}
    return s.per, combine(s, models, cfg, log)
