from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .models import Ensemble, average
from .multirate import MultiRate, twin_name
from .prediction import PredictCfg
from .scoring import Scored, score_model

TIMING = ("train_s", "ms_per_forecast")


def _average(s: Scored, name: str, names: list[str]) -> None:
    for p in s.prepared:
        held, test = (average([s.made[m][p.config][k] for m in names]) for k in (0, 1))
        s.add(p, name, held, test, {k: round(sum(s.per[p.config][m][k] for m in names), 2) for k in TIMING})


def _ensemble(s: Scored, models: list[Any], name: str) -> dict[str, Any]:
    if len(models) < 2:
        return {}
    _average(s, name, [m.name for m in models])
    return {name: Ensemble(models, name)}


def _twins(s: Scored, ours: list[Any], rates: list[int], log: Callable[[str], None]) -> dict[str, Any]:
    twins = [MultiRate(m, rates) for m in ours]
    for twin in twins:
        score_model(twin, s, s.per[s.prepared[0].config][twin.model.name]["train_s"], log)
    return {t.name: t for t in twins} | _ensemble(s, twins, twin_name(Ensemble.name))


def combine(s: Scored, ours: list[Any], cfg: PredictCfg, log: Callable[[str], None]) -> dict[str, Any]:
    trained = {m.name: m for m in ours} | _ensemble(s, ours, Ensemble.name)
    if cfg.multi_rate:
        trained |= _twins(s, ours, cfg.rates, log)
    return trained
