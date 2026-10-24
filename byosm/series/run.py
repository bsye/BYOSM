from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from .gifteval import DOMAINS, horizon, load, season, splits, windows
from .metrics import geomean, relative, summary
from .models import Boosted, rivals
from .patch import PatchModel, PretrainedPatch, weights_path

REPORT = "series:"
NAIVE = "seasonal naive"
SERIES_EVENT = "series"


def run_config(config: str, log: Callable[[str], None]) -> dict[str, Any]:
    series = load(config)
    h, s = horizon(series[0].freq), season(series[0].freq)
    tests = splits(series)
    cut = windows(series) * h
    histories = [x.values[: len(x.values) - cut] for x in series]
    log(f"  {config}: {len(series)} series, {len(tests)} windows of {h} steps")
    out = {}
    for model in [*rivals(), Boosted(), PatchModel(), *([PretrainedPatch()] if weights_path().exists() else [])]:
        t0 = time.perf_counter()
        model.fit(histories, s, h)
        trained = time.perf_counter() - t0
        t1 = time.perf_counter()
        many = getattr(model, "predict_many", None)
        forecasts = many([w["history"] for w in tests], h) if many else [model.predict(w["history"], h) for w in tests]
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


def _overall(per: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    overall: dict[str, dict[str, Any]] = {}
    for m in sorted(set.intersection(*(set(r) for r in per.values()))):
        rel = [relative({k: r[m][k] for k in ("mase", "crps")}, r[NAIVE]) for r in per.values()]
        overall[m] = {k: geomean([x[k] for x in rel]) for k in ("mase", "crps")}
        for k in ("train_s", "ms_per_forecast"):
            if all(k in r[m] for r in per.values()):
                overall[m][k] = round(float(np.mean([r[m][k] for r in per.values()])), 3)
        overall[m]["published"] = any(r[m].get("published") for r in per.values())
    return overall


def run_domain(ws: Path, domain: str, log: Callable[[str], None] = print) -> dict[str, Any]:
    from ..db import get_or_add, session
    from ..db.models import WorkspaceReport
    from ..events import emit

    configs, per = DOMAINS.get(domain, [domain]), {}
    for config in configs:
        per[config] = run_config(config, log)
        pending = [c for c in configs if c not in per]
        result = {"domain": domain, "at": time.time(), "configs": per, "overall": _overall(per), "pending": pending}
        with session(ws) as s:
            get_or_add(s, WorkspaceReport, kind=REPORT + domain).data = result
        emit(SERIES_EVENT, domain=domain, config=config, pending=len(pending))
    for m, score in result["overall"].items():
        log(f"  {domain} {m}: MASE {score['mase']:.3f} and CRPS {score['crps']:.3f} relative to seasonal naive")
    return result


def _with_published(report: dict[str, Any]) -> dict[str, Any]:
    from .published import scores

    per = {c: {**r, **scores(c, r[NAIVE]["mase"])} for c, r in report.get("configs", {}).items()}
    return {**report, "configs": per, "overall": _overall(per)} if per else report


def reports(ws: Path) -> list[dict[str, Any]]:
    from sqlalchemy import select

    from ..db import session
    from ..db.models import WorkspaceReport

    with session(ws) as s:
        rows = s.scalars(select(WorkspaceReport.data).where(WorkspaceReport.kind.startswith(REPORT))).all()
    found = {r["domain"]: _with_published(r) for r in rows if r}
    return [found.get(d) or {"domain": d, "at": None} for d in [*DOMAINS, *(k for k in found if k not in DOMAINS)]]
