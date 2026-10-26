from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from .gifteval import horizon, load, season, splits, windows
from .metrics import summary
from .models import Boosted, Ensemble, Group, average, rivals
from .patch import PatchModel, PretrainedPatch, pretrained
from .specialist import is_rival

ENSEMBLE = Ensemble.name
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


def candidates(effort: int) -> list[Any]:
    base = [PretrainedPatch(effort)] if pretrained() else []
    return [*rivals(), Boosted(effort), PatchModel(effort), *base]


def _forecast(model: Any, windows: list[dict[str, Any]], p: Prepared) -> list[np.ndarray]:
    return model.predict_many([w["history"] for w in windows], p.season, p.horizon)


def _scores(p: Prepared, held: list[np.ndarray], test: list[np.ndarray]) -> dict[str, Any]:
    a = summary(p.calib, held, [p.season] * len(p.calib))
    b = summary(p.tests, test, [p.season] * len(p.tests))
    return {**b, "calib_mase": a["mase"], "calib_crps": a["crps"]}


def _line(name: str, score: dict[str, Any]) -> str:
    return f"    {name}: calib MASE {score['calib_mase']:.3f}, test MASE {score['mase']:.3f}, CRPS {score['crps']:.3f}, {score['ms_per_forecast']:.1f} ms per forecast"


def fit_domain(
    configs: list[str], effort: int, log: Callable[[str], None], progress: Progress
) -> tuple[dict[str, Any], dict[str, Any]]:
    prepared = [prepare(c, log) for c in configs]
    groups = [Group(p.histories, p.season, p.horizon) for p in prepared]
    models, per, made = candidates(effort), {p.config: {} for p in prepared}, {}
    for i, model in enumerate(models):
        t0 = time.perf_counter()
        model.fit(groups)
        trained = time.perf_counter() - t0
        log(f"  {model.name}: one model for {len(prepared)} datasets, trained in {trained:.1f} s")
        for p in prepared:
            held, t1 = _forecast(model, p.calib, p), time.perf_counter()
            test = _forecast(model, p.tests, p)
            ms = (time.perf_counter() - t1) / max(1, len(p.tests)) * 1000
            per[p.config][model.name] = _scores(p, held, test) | {
                "train_s": round(trained, 2),
                "ms_per_forecast": round(ms, 2),
            }
            made.setdefault(model.name, {})[p.config] = (held, test)
            log(f"  {p.config}" + _line(model.name, per[p.config][model.name]))
        progress(per, [m.name for m in models[i + 1 :]])
    ours = [m for m in made if not is_rival(m)]
    if len(ours) > 1:
        for p in prepared:
            held, test = (average([made[m][p.config][k] for m in ours]) for k in (0, 1))
            timing = {k: round(sum(per[p.config][m][k] for m in ours), 2) for k in ("train_s", "ms_per_forecast")}
            per[p.config][ENSEMBLE] = _scores(p, held, test) | timing
    trained = {m.name: m for m in models if m.name in ours}
    return per, {**trained, ENSEMBLE: Ensemble(list(trained.values()))}
