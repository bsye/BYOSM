from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from .gifteval import DOMAINS, horizon, load, season, splits, windows
from .metrics import geomean, relative, summary
from .models import Boosted, rivals

REPORT = "series:"
NAIVE = "seasonal naive"


def run_config(config: str, log: Callable[[str], None]) -> dict[str, Any]:
    series = load(config)
    h, s = horizon(series[0].freq), season(series[0].freq)
    tests = splits(series)
    cut = windows(series) * h
    histories = [x.values[: len(x.values) - cut] for x in series]
    log(f"  {config}: {len(series)} series, {len(tests)} windows of {h} steps")
    out = {}
    for model in [*rivals(), Boosted()]:
        t0 = time.perf_counter()
        model.fit(histories, s, h)
        trained = time.perf_counter() - t0
        t1 = time.perf_counter()
        forecasts = [model.predict(w["history"], h) for w in tests]
        per = (time.perf_counter() - t1) / max(1, len(tests)) * 1000
        out[model.name] = {
            **summary(tests, forecasts, [s] * len(tests)),
            "train_s": round(trained, 2),
            "ms_per_forecast": round(per, 2),
        }
        log(
            f"    {model.name}: MASE {out[model.name]['mase']:.3f}, CRPS {out[model.name]['crps']:.3f}, trained in {trained:.1f} s, {per:.1f} ms per forecast"
        )
    return out


def run_domain(ws: Path, domain: str, log: Callable[[str], None] = print) -> dict[str, Any]:
    from ..db import get_or_add, session
    from ..db.models import WorkspaceReport

    configs = DOMAINS.get(domain, [domain])
    per = {c: run_config(c, log) for c in configs}
    models = sorted({m for r in per.values() for m in r})
    overall = {}
    for m in models:
        rel = [relative({k: r[m][k] for k in ("mase", "crps")}, r[NAIVE]) for r in per.values() if m in r]
        overall[m] = {k: geomean([x[k] for x in rel]) for k in ("mase", "crps")}
        overall[m] |= {
            k: round(float(np.mean([r[m][k] for r in per.values() if m in r])), 3)
            for k in ("train_s", "ms_per_forecast")
        }
        log(
            f"  {domain} {m}: MASE {overall[m]['mase']:.3f} and CRPS {overall[m]['crps']:.3f} relative to seasonal naive"
        )
    result = {"domain": domain, "at": time.time(), "configs": per, "overall": overall}
    with session(ws) as s:
        get_or_add(s, WorkspaceReport, kind=REPORT + domain).data = result
    return result
