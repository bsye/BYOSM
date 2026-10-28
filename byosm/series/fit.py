from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

from ..events import emit
from ..lab.owners import Owner
from ..lab.plugin import Method
from .combine import combine
from .models import Boosted, Group, QuantileBoosted, SeasonalNaive, rivals
from .patch import PatchModel, PretrainedPatch, pretrained
from .prediction import FitCfg, Options
from .scoring import Progress, Scored, score_model, scored
from .specialist import is_rival

FITTING_EVENT = "fitting"
Log = Callable[[str], None]


def candidates(effort: int, fit: FitCfg, lab: Owner | None = None) -> list[Any]:
    method = Method(lab, fit.lab_method) if fit.lab_method else None
    base = [PretrainedPatch(effort, fit.recipe, method)] if pretrained(fit.recipe) else []
    boosted = QuantileBoosted(effort) if fit.quantile_boosting else Boosted(effort)
    return [*rivals(), boosted, PatchModel(effort, method), *base]


def refreshed(
    saved: list[Any], effort: int, before: FitCfg, after: FitCfg, lab: Owner | None = None
) -> tuple[list[Any], list[Any]]:
    old = {m.name for m in candidates(effort, before, lab)}
    new = [m for m in candidates(effort, after, lab) if not is_rival(m.name)]
    replaced = old - {m.name for m in new}
    return [m for m in saved if m.name not in replaced], [m for m in new if m.name not in old]


def _groups(s: Scored) -> list[Group]:
    return [Group(p.histories, p.season, p.horizon, p.clocks) for p in s.prepared]


def _train(model: Any, s: Scored, groups: list[Group], log: Log) -> None:
    t0 = time.perf_counter()
    model.fit(groups)
    took = time.perf_counter() - t0
    log(f"  {model.name}: one model for {len(s.prepared)} datasets, trained in {took:.1f} s")
    score_model(model, s, took, log)


def fit_domain(
    configs: list[str], effort: int, log: Log, progress: Progress, options: Options, lab: Owner | None = None
) -> tuple[dict[str, Any], dict[str, Any]]:
    s = scored(configs, log)
    groups, models = _groups(s), candidates(effort, options.fit, lab)
    for i, model in enumerate(models):
        log(f"  training {model.name}, recipe {i + 1} of {len(models)}")
        emit(FITTING_EVENT, recipe=model.name, index=i + 1, of=len(models))
        _train(model, s, groups, log)
        progress(s.per, [m.name for m in models[i + 1 :]])
    return s.per, combine(s, [m for m in models if not is_rival(m.name)], options.predict, log)


def _train_s(parent: dict[str, dict[str, Any]], name: str) -> float:
    return next((r[name]["train_s"] for r in parent.values() if "train_s" in r.get(name, {})), 0.0)


def rescore(
    configs: list[str],
    models: tuple[list[Any], list[Any]],
    parent: dict[str, dict[str, Any]],
    log: Log,
    progress: Progress,
    options: Options,
) -> tuple[dict[str, Any], dict[str, Any]]:
    (kept, fresh), s = models, scored(configs, log)
    every, groups = [SeasonalNaive(), *kept, *fresh], _groups(s)
    for i, model in enumerate(every):
        if any(model is f for f in fresh):
            log(f"  training {model.name}, the only model this option changes")
            _train(model, s, groups, log)
        else:
            score_model(model, s, _train_s(parent, model.name), log)
        progress(s.per, [m.name for m in every[i + 1 :]])
    for config, row in s.per.items():
        old = parent.get(config, {}).items()
        row |= {m: r for m, r in old if is_rival(m) and m not in row and not r.get("published")}
    return s.per, combine(s, [*kept, *fresh], options.predict, log)
