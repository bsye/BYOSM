from __future__ import annotations

import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from .fit import fit_domain
from .gifteval import DOMAINS
from .metrics import geomean, ratio_interval
from .specialist import CI_DRAWS, CI_SEED, NAIVE, add_version, choose
from .specialist import summary as forecaster

REPORT = "series:"
SERIES_EVENT = "series"
MIN_GAIN = 0.01
SCORES = ("mase", "crps", "calib_mase", "calib_crps")


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
    kept = {
        "versions": previous.get("versions", []),
        "promoted": previous.get("promoted"),
        "created": previous.get("created") or datetime.now(UTC).isoformat(),
    }
    effort = len(kept["versions"])
    log(f"  effort {effort}: {2**effort}x the base training steps")

    def progress(per: dict[str, dict[str, Any]], pending: list[str]) -> None:
        _save(
            ws,
            {
                "domain": domain,
                "at": time.time(),
                "configs": _stored(per),
                "overall": _overall(per),
                "pending": pending,
                **kept,
            },
        )
        emit(SERIES_EVENT, domain=domain, pending=len(pending))

    per = fit_domain(DOMAINS.get(domain, [domain]), effort, log, progress)
    report = {
        "domain": domain,
        "at": time.time(),
        "configs": _stored(per),
        "overall": _overall(per),
        "pending": [],
        **kept,
    }
    report = add_version(report, _interval(per, choose(report["overall"])), effort)
    _save(ws, report)
    if report.get("versions"):
        newest = report["versions"][-1]
        log(f"  {newest['name']}: {newest['recipe']}, promoted {report['promoted']}")
    return report


def _with_published(report: dict[str, Any]) -> dict[str, Any]:
    from .published import scores

    def merged(configs: dict[str, Any]) -> dict[str, Any]:
        return {c: {**r, **scores(c, r[NAIVE]["mase"])} for c, r in configs.items()}

    per = merged(report.get("configs", {}))
    versions = [{**v, "configs": merged(v["configs"])} if v.get("configs") else v for v in report.get("versions", [])]
    return {**report, "configs": per, "overall": _overall(per), "versions": versions} if per else report


def promoted_board(report: dict[str, Any]) -> dict[str, Any]:
    version = next((v for v in report.get("versions", []) if v["name"] == report.get("promoted")), None)
    configs = (version or {}).get("configs")
    return {"configs": configs, "overall": _overall(configs)} if configs else {}


def reports(ws: Path) -> list[dict[str, Any]]:
    from sqlalchemy import select

    from ..db import session
    from ..db.models import WorkspaceReport

    with session(ws) as s:
        rows = s.scalars(select(WorkspaceReport.data).where(WorkspaceReport.kind.startswith(REPORT))).all()
    found = {r["domain"]: _with_published(r) for r in rows if r}
    return [found.get(d) or {"domain": d, "at": None} for d in [*DOMAINS, *(k for k in found if k not in DOMAINS)]]


def _gain(report: dict[str, Any]) -> float:
    versions = report.get("versions", [])
    if len(versions) < 2:
        return 1.0
    before = min(v["calib"] for v in versions[:-1])
    return (before - versions[-1]["calib"]) / before


def until_rival(ws: Path, domain: str, rounds: int, log: Callable[[str], None] = print) -> dict[str, Any]:
    from .patch import weights_path

    for n in range(rounds):
        report = run_domain(ws, domain, log)
        rival = forecaster(_with_published(report))["rival"]
        if rival and rival["ours"] < rival["score"]:
            log(f"  beat {rival['name']} after {n + 1} round(s): {rival['ours']:.3f} vs {rival['score']:.3f}")
            return report
        log(f"  round {n + 1}: hardest rival {rival}")
        if _gain(report) < MIN_GAIN:
            lever = "a bigger base model" if weights_path().exists() else "the pretrained base (byosm series pretrain)"
            log(f"  more training gained under {MIN_GAIN:.0%} on calib: stopping, the next lever is {lever}")
            return report
    return report
