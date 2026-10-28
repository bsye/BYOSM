from __future__ import annotations

import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .board import interval, overall, reports, save_report, stored, with_published
from .combine import members
from .escalation import base_for, escalate, lever, recipe_of, upstream
from .fit import fit_domain, refreshed, rescore
from .gifteval import DOMAINS
from .prediction import Options, next_technique, of
from .saved import load_model, save, saved
from .scoring import Progress
from .specialist import PLATEAU, add_version, choose, promoted_version
from .specialist import summary as forecaster

SERIES_EVENT = "series"
MIN_GAIN = 0.01
Log = Callable[[str], None]


def _previous(ws: Path, domain: str) -> dict[str, Any]:
    return next((r for r in reports(ws) if r["domain"] == domain), {})


def _kept(previous: dict[str, Any]) -> dict[str, Any]:
    return {
        "versions": previous.get("versions", []),
        "promoted": previous.get("promoted"),
        "created": previous.get("created") or datetime.now(UTC).isoformat(),
    }


def _report(domain: str, per: dict[str, Any], pending: list[str], kept: dict[str, Any]) -> dict[str, Any]:
    return {
        "domain": domain,
        "at": time.time(),
        "configs": stored(per),
        "overall": overall(per),
        "pending": pending,
        **kept,
    }


def _progress(ws: Path, domain: str, kept: dict[str, Any]) -> Progress:
    from ..events import emit

    def progress(per: dict[str, dict[str, Any]], pending: list[str]) -> None:
        save_report(ws, _report(domain, per, pending, kept))
        emit(SERIES_EVENT, domain=domain, pending=len(pending))

    return progress


def _finish(ws: Path, domain: str, made: tuple[dict, dict], kept: dict, at: tuple[int, Options], log: Log) -> dict:
    (per, trained), (effort, options) = made, at
    report = _report(domain, per, [], kept)
    report = add_version(report, interval(per, choose(report["overall"])), effort, options.model_dump())
    save_report(ws, report)
    if report.get("versions"):
        newest = report["versions"][-1]
        save(ws, domain, newest["name"], trained[newest["recipe"]])
        log(f"  {newest['name']}: {newest['recipe']}, promoted {report['promoted']}")
    return report


def run_domain(ws: Path, domain: str, log: Log = print) -> dict[str, Any]:
    previous = _previous(ws, domain)
    best = promoted_version(previous)
    effort, options = (best["effort"] + 1 if best else 0), of(best)
    log(f"  effort {effort}: {2**effort}x the base training steps")
    kept = _kept(previous)
    made = fit_domain(DOMAINS.get(domain, [domain]), effort, log, _progress(ws, domain, kept), options)
    return _finish(ws, domain, made, kept, (effort, options), log)


def apply_technique(ws: Path, domain: str, options: Options, models: list[Any], log: Log = print) -> dict[str, Any]:
    previous = _previous(ws, domain)
    best = promoted_version(previous)
    kept = {**_kept(previous), PLATEAU: previous.get(PLATEAU)}
    parent, before = best.get("configs") or {}, of(best).fit
    base_for(before.recipe, options.fit.recipe, log)
    split = refreshed(models, best["effort"], before, options.fit)
    made = rescore(DOMAINS.get(domain, [domain]), split, parent, log, _progress(ws, domain, kept), options)
    return _finish(ws, domain, made, kept, (best["effort"], options), log)


def _gain(report: dict[str, Any]) -> float:
    versions = report.get("versions", [])
    if len(versions) < 2:
        return 1.0
    before = min(v["calib"] for v in versions[:-1])
    return (before - versions[-1]["calib"]) / before


def _mark(ws: Path, report: dict[str, Any], plateau: dict[str, Any] | None) -> dict[str, Any]:
    report = {**report, PLATEAU: plateau}
    save_report(ws, report)
    return report


def _technique(ws: Path, domain: str, report: dict[str, Any]) -> tuple[str, Options, list[Any]] | None:
    found, best = next_technique(report), promoted_version(report)
    if not found or not saved(ws, domain, best["name"]):
        return None
    models = members(load_model(ws, domain, best["name"]))
    return (*found, models) if models else None


def _round(ws: Path, domain: str, report: dict[str, Any], log: Log) -> tuple[dict[str, Any] | None, bool]:
    found = _technique(ws, domain, report)
    if found:
        technique, options, models = found
        log(f"  {technique}: trying it on the models of {report['promoted']}, training only what it changes")
        return apply_technique(ws, domain, options, models, log), False
    stuck = report.get(PLATEAU)
    if stuck and stuck["upstream"] == upstream(recipe_of(report)) and not escalate(report, log):
        return None, True
    return run_domain(ws, domain, log), True


def until_rival(ws: Path, domain: str, rounds: int, log: Log = print) -> dict[str, Any]:
    report = _previous(ws, domain)
    for n in range(rounds):
        found, trained = _round(ws, domain, report, log)
        if found is None:
            return report
        report = found
        rival = forecaster(with_published(report))["rival"]
        if rival and rival["ours"] < rival["score"]:
            log(f"  beat {rival['name']} after {n + 1} round(s): {rival['ours']:.3f} vs {rival['score']:.3f}")
            return _mark(ws, report, None)
        log(f"  round {n + 1}: hardest rival {rival}")
        if trained and _gain(report) < MIN_GAIN:
            recipe, effort = recipe_of(report), report["versions"][-1]["effort"]
            log(f"  more training gained under {MIN_GAIN:.0%} on calib: stopping, the next lever is {lever(recipe)}")
            return _mark(ws, report, {"effort": effort, "upstream": upstream(recipe), "lever": lever(recipe)})
    return report
