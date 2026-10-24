from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from .gifteval import DOMAINS, horizon, load, season, splits, windows
from .metrics import geomean, ratio_interval, summary
from .models import Boosted, rivals
from .patch import PatchModel, PretrainedPatch, weights_path
from .specialist import CI_DRAWS, CI_SEED, NAIVE, add_version, choose

REPORT = "series:"
SERIES_EVENT = "series"
SCORES = ("mase", "crps", "calib_mase", "calib_crps")


def _forecast(model: Any, windows: list[dict[str, Any]], h: int) -> list[np.ndarray]:
    many = getattr(model, "predict_many", None)
    return many([w["history"] for w in windows], h) if many else [model.predict(w["history"], h) for w in windows]


def run_config(config: str, log: Callable[[str], None]) -> dict[str, Any]:
    series = load(config)
    h, s = horizon(series[0].freq), season(series[0].freq)
    tests, calib = splits(series), splits(series, back=1)
    cut = 2 * windows(series) * h
    histories = [x.values[: len(x.values) - cut] for x in series]
    log(f"  {config}: {len(series)} series, {len(calib)} calib and {len(tests)} test windows of {h} steps")
    out = {}
    for model in [*rivals(), Boosted(), PatchModel(), *([PretrainedPatch()] if weights_path().exists() else [])]:
        t0 = time.perf_counter()
        model.fit(histories, s, h)
        trained = time.perf_counter() - t0
        held = summary(calib, _forecast(model, calib, h), [s] * len(calib))
        t1 = time.perf_counter()
        test = summary(tests, _forecast(model, tests, h), [s] * len(tests))
        per = (time.perf_counter() - t1) / max(1, len(tests)) * 1000
        out[model.name] = {
            **test,
            "calib_mase": held["mase"],
            "calib_crps": held["crps"],
            "train_s": round(trained, 2),
            "ms_per_forecast": round(per, 2),
        }
        log(
            f"    {model.name}: calib MASE {held['mase']:.3f}, test MASE {test['mase']:.3f}, CRPS {test['crps']:.3f}, trained in {trained:.1f} s, {per:.1f} ms per forecast"
        )
    return out


def _overall(per: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    overall: dict[str, dict[str, Any]] = {}
    for m in sorted(set.intersection(*(set(r) for r in per.values()))):
        overall[m] = {}
        for k in SCORES:
            if all(k in r[m] and k in r[NAIVE] for r in per.values()):
                overall[m][k] = geomean([r[m][k] / r[NAIVE][k] for r in per.values()])
        for k in ("train_s", "ms_per_forecast"):
            if all(k in r[m] for r in per.values()):
                overall[m][k] = round(float(np.mean([r[m][k] for r in per.values()])), 3)
        overall[m]["published"] = any(r[m].get("published") for r in per.values())
    return overall


def _interval(per: dict[str, dict[str, Any]], recipe: str | None) -> list[float] | None:
    if recipe is None:
        return None
    ours, base = [r[recipe]["windows"] for r in per.values()], [r[NAIVE]["windows"] for r in per.values()]
    return ratio_interval(ours, base, CI_DRAWS, CI_SEED)


def _stored(per: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {c: {m: {k: v for k, v in s.items() if k != "windows"} for m, s in r.items()} for c, r in per.items()}


def _save(ws: Path, report: dict[str, Any]) -> None:
    from ..db import get_or_add, session
    from ..db.models import WorkspaceReport

    with session(ws) as s:
        get_or_add(s, WorkspaceReport, kind=REPORT + report["domain"]).data = report


def run_domain(ws: Path, domain: str, log: Callable[[str], None] = print) -> dict[str, Any]:
    from ..events import emit

    previous = next((r for r in reports(ws) if r["domain"] == domain), {})
    kept = {"versions": previous.get("versions", []), "promoted": previous.get("promoted")}
    configs, per = DOMAINS.get(domain, [domain]), {}
    for config in configs:
        per[config] = run_config(config, log)
        pending = [c for c in configs if c not in per]
        report = {"domain": domain, "at": time.time(), "configs": _stored(per), "overall": _overall(per), **kept}
        _save(ws, {**report, "pending": pending})
        emit(SERIES_EVENT, domain=domain, config=config, pending=len(pending))
    report = add_version({**report, "pending": []}, _interval(per, choose(report["overall"])))
    _save(ws, report)
    for m, score in report["overall"].items():
        log(f"  {domain} {m}: test MASE {score['mase']:.3f}, calib MASE {score.get('calib_mase', 0):.3f} vs naive")
    if report.get("versions"):
        newest = report["versions"][-1]
        log(f"  {newest['name']}: {newest['recipe']}, promoted {report['promoted']}")
    return report


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
