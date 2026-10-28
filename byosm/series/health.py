from __future__ import annotations

from pathlib import Path
from typing import Any

from ..health_score import FAIL, OK, UNKNOWN, WARN, check, report
from .specialist import NAIVE, best_rival, promoted_version
from .steps import keep_training, write_method

AGREE_OK, AGREE_WARN = 0.10, 0.25
WIDTH_OK, WIDTH_WARN = 0.05, 0.15
RIVAL_NEAR = 0.05
LAB_METHOD = "lab-method"


def _naive(configs: dict[str, Any], ours: str) -> dict:
    rows = [r for r in configs.values() if ours in r and NAIVE in r]
    wins = sum(r[ours]["mase"] < r[NAIVE]["mase"] for r in rows)
    why = "The promoted version should forecast better than repeating the last season on every dataset."
    if not rows:
        return check("robust", "Beats seasonal naive everywhere", 2, UNKNOWN, "not scored per dataset", why)
    status = OK if wins == len(rows) else WARN if wins * 2 >= len(rows) else FAIL
    return check("robust", "Beats seasonal naive everywhere", 2, status, f"{wins} of {len(rows)} datasets", why)


def _gaps(version: dict[str, Any]) -> dict[str, float]:
    ours, out = version["recipe"], {}
    for config, r in (version.get("configs") or {}).items():
        if ours in r and NAIVE in r and r[ours].get("calib_mase") and r[NAIVE].get("calib_mase"):
            calib = r[ours]["calib_mase"] / r[NAIVE]["calib_mase"]
            out[config] = abs(r[ours]["mase"] / r[NAIVE]["mase"] - calib) / calib
    return out


def _agreement(version: dict[str, Any]) -> dict:
    gap = abs(version["test"] - version["calib"]) / version["calib"]
    worst = max(_gaps(version).items(), key=lambda kv: kv[1], default=None)
    where = ""
    if worst and worst[1] > gap:
        where, gap = f" on {worst[0]}", worst[1]
    status = OK if gap <= AGREE_OK else WARN if gap <= AGREE_WARN else FAIL
    why = "The recipe is picked on the calib windows; a large gap to test on any dataset means the pick does not carry over there."
    return check("overfit", "Calib and test agree", 1, status, f"{gap:.0%} apart{where}", why)


def _certainty(version: dict[str, Any]) -> dict:
    ci = version.get("ci")
    why = "Width of the 95% interval of the test score relative to seasonal naive: narrow means it can be trusted."
    if not ci:
        return check("certainty", "Precise score", 1, UNKNOWN, "no interval", why)
    width = ci[1] - ci[0]
    status = OK if width <= WIDTH_OK else WARN if width <= WIDTH_WARN else FAIL
    return check("certainty", "Precise score", 1, status, f"±{width / 2:.3f}", why)


def _next_fix(report_: dict[str, Any]) -> dict[str, Any]:
    from ..improve.techniques import techniques
    from .prediction import next_technique

    titles, found, domain = {t.id: t.title for t in techniques()}, next_technique(report_), report_["domain"]
    if found:
        return {"techniques": [titles[found[0]]], "args": keep_training(domain)}
    return {"techniques": [titles[LAB_METHOD]], "args": write_method(domain)}


def _rival(report_: dict[str, Any]) -> dict:
    rival = best_rival(report_)
    why = "The hardest rival on the datasets both were scored on, as MASE relative to seasonal naive."
    if not rival:
        return check("rival", "Beats the best rival", 3, UNKNOWN, "no rival scored", why)
    ahead = rival["score"] - rival["ours"]
    status = OK if ahead > 0 else WARN if ahead > -RIVAL_NEAR else FAIL
    found = check("rival", "Beats the best rival", 3, status, f"{ahead:+.3f} vs {rival['name']}", why)
    return found if status == OK else {**found, **_next_fix(report_)}


def health(report_: dict[str, Any]) -> dict[str, Any] | None:
    version = promoted_version(report_)
    if not version:
        return None
    configs = version.get("configs") or {}
    return report([_rival(report_), _naive(configs, version["recipe"]), _agreement(version), _certainty(version)])


def of_domain(ws: Path, name: str) -> dict[str, Any] | None:
    from .board import reports

    found = next((r for r in reports(ws) if r["domain"] == name and r["at"]), None)
    return health(found) if found else None
