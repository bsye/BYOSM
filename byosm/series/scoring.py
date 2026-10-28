from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .gifteval import horizon, load, season, splits, windows
from .metrics import summary

Forecasts = list[np.ndarray]
Progress = Callable[[dict[str, dict[str, Any]], list[str]], None]


@dataclass
class Prepared:
    config: str
    season: int
    horizon: int
    calib: list[dict[str, Any]]
    tests: list[dict[str, Any]]
    histories: list[np.ndarray]


def prepare(config: str, log: Callable[[str], None]) -> Prepared:
    series = load(config)
    h, s = horizon(series[0].freq), season(series[0].freq)
    cut = 2 * windows(series) * h
    p = Prepared(
        config, s, h, splits(series, back=1), splits(series), [x.values[: len(x.values) - cut] for x in series]
    )
    log(f"  {config}: {len(series)} series, {len(p.calib)} calib and {len(p.tests)} test windows of {h} steps")
    return p


def scores(p: Prepared, held: Forecasts, test: Forecasts) -> dict[str, Any]:
    a = summary(p.calib, held, [p.season] * len(p.calib))
    b = summary(p.tests, test, [p.season] * len(p.tests))
    return {**b, "calib_mase": a["mase"], "calib_crps": a["crps"]}


@dataclass
class Scored:
    prepared: list[Prepared]
    made: dict[str, dict[str, tuple[Forecasts, Forecasts]]] = field(default_factory=dict)
    per: dict[str, dict[str, dict[str, Any]]] = field(init=False)

    def __post_init__(self) -> None:
        self.per = {p.config: {} for p in self.prepared}

    def add(self, p: Prepared, name: str, held: Forecasts, test: Forecasts, extra: dict[str, Any]) -> None:
        self.per[p.config][name] = scores(p, held, test) | extra
        self.made.setdefault(name, {})[p.config] = (held, test)


def scored(configs: list[str], log: Callable[[str], None]) -> Scored:
    return Scored([prepare(c, log) for c in configs])


def _line(name: str, score: dict[str, Any]) -> str:
    return f"    {name}: calib MASE {score['calib_mase']:.3f}, test MASE {score['mase']:.3f}, CRPS {score['crps']:.3f}, {score['ms_per_forecast']:.1f} ms per forecast"


def _forecast(model: Any, windows: list[dict[str, Any]], p: Prepared) -> Forecasts:
    return model.predict_many([w["history"] for w in windows], p.season, p.horizon)


def score_model(model: Any, s: Scored, train_s: float, log: Callable[[str], None]) -> None:
    for p in s.prepared:
        held, t1 = _forecast(model, p.calib, p), time.perf_counter()
        test = _forecast(model, p.tests, p)
        ms = (time.perf_counter() - t1) / max(1, len(p.tests)) * 1000
        s.add(p, model.name, held, test, {"train_s": round(train_s, 2), "ms_per_forecast": round(ms, 2)})
        log(f"  {p.config}" + _line(model.name, s.per[p.config][model.name]))
