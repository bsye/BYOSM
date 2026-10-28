from __future__ import annotations

from typing import Any

from .models import Ensemble, average
from .scoring import Scored

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


def combine(s: Scored, ours: list[Any]) -> dict[str, Any]:
    return {m.name: m for m in ours} | _ensemble(s, ours, Ensemble.name)
