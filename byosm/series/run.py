from __future__ import annotations

import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .board import interval, overall, reports, save_report, stored, with_published
from .fit import fit_domain
from .gifteval import DOMAINS
from .saved import save
from .scoring import Progress
from .specialist import add_version, choose, promoted_version
from .specialist import summary as forecaster

SERIES_EVENT = "series"
MIN_GAIN = 0.01
PLATEAU = "plateau"
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


def _finish(ws: Path, domain: str, made: tuple[dict, dict], kept: dict, effort: int, log: Log) -> dict[str, Any]:
    per, trained = made
    report = _report(domain, per, [], kept)
    report = add_version(report, interval(per, choose(report["overall"])), effort)
    save_report(ws, report)
    if report.get("versions"):
        newest = report["versions"][-1]
        save(ws, domain, newest["name"], trained[newest["recipe"]])
        log(f"  {newest['name']}: {newest['recipe']}, promoted {report['promoted']}")
    return report


def run_domain(ws: Path, domain: str, log: Log = print) -> dict[str, Any]:
    previous = _previous(ws, domain)
    best = promoted_version(previous)
    effort = best["effort"] + 1 if best else 0
    log(f"  effort {effort}: {2**effort}x the base training steps")
    kept = _kept(previous)
    made = fit_domain(DOMAINS.get(domain, [domain]), effort, log, _progress(ws, domain, kept))
    return _finish(ws, domain, made, kept, effort, log)


def _gain(report: dict[str, Any]) -> float:
    versions = report.get("versions", [])
    if len(versions) < 2:
        return 1.0
    before = min(v["calib"] for v in versions[:-1])
    return (before - versions[-1]["calib"]) / before


def _upstream() -> dict[str, Any]:
    from .corpus import files
    from .patch import pretrained

    return {"base": (pretrained() or (None,))[0], "corpus": len(files())}


def _lever() -> str:
    from .patch import missing_size

    size = missing_size()
    return f"the {size} base (series pretrain --size {size})" if size else "more pretraining data (series corpus --gb)"


def _mark(ws: Path, report: dict[str, Any], plateau: dict[str, Any] | None) -> dict[str, Any]:
    report = {**report, PLATEAU: plateau}
    save_report(ws, report)
    return report


def _escalate(ws: Path, previous: dict[str, Any], log: Log) -> bool:
    from .corpus import PRETRAIN_STEPS
    from .patch import missing_size
    from .pretrain import pretrain

    plateau = previous[PLATEAU]
    size = missing_size()
    if not size:
        log(
            f"  more training stopped helping at effort {plateau['effort']} and every base size is pretrained: "
            "the next lever is more pretraining data (series corpus --gb), so no new version is trained"
        )
        return False
    log(f"  more training stopped helping at effort {plateau['effort']}: pretraining the {size} base first")
    pretrain(log, PRETRAIN_STEPS, size)
    return True


def until_rival(ws: Path, domain: str, rounds: int, log: Log = print) -> dict[str, Any]:
    previous = _previous(ws, domain)
    stuck = previous.get(PLATEAU)
    if stuck and stuck["upstream"] == _upstream() and not _escalate(ws, previous, log):
        return previous
    report = previous
    for n in range(rounds):
        report = run_domain(ws, domain, log)
        rival = forecaster(with_published(report))["rival"]
        if rival and rival["ours"] < rival["score"]:
            log(f"  beat {rival['name']} after {n + 1} round(s): {rival['ours']:.3f} vs {rival['score']:.3f}")
            return _mark(ws, report, None)
        log(f"  round {n + 1}: hardest rival {rival}")
        if _gain(report) < MIN_GAIN:
            log(f"  more training gained under {MIN_GAIN:.0%} on calib: stopping, the next lever is {_lever()}")
            effort = report["versions"][-1]["effort"]
            return _mark(ws, report, {"effort": effort, "upstream": _upstream(), "lever": _lever()})
    return report
