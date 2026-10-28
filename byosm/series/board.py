from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .gifteval import DOMAINS
from .metrics import geomean, ratio_interval
from .specialist import CI_DRAWS, CI_SEED, NAIVE, promoted_version

REPORT = "series:"
SCORES = ("mase", "crps", "calib_mase", "calib_crps")


def overall(per: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for m in sorted({m for r in per.values() for m in r}):
        rows = [r for r in per.values() if m in r and NAIVE in r]
        out[m] = {"datasets": len(rows)}
        for k in SCORES:
            if rows and all(k in r[m] and k in r[NAIVE] for r in rows):
                out[m][k] = geomean([r[m][k] / r[NAIVE][k] for r in rows])
        for k in ("train_s", "ms_per_forecast"):
            if rows and all(k in r[m] for r in rows):
                out[m][k] = round(float(np.mean([r[m][k] for r in rows])), 3)
        out[m]["published"] = any(r[m].get("published") for r in rows)
    return out


def interval(per: dict[str, dict[str, Any]], recipe: str | None) -> list[float] | None:
    if recipe is None:
        return None
    ours, base = [r[recipe]["windows"] for r in per.values()], [r[NAIVE]["windows"] for r in per.values()]
    return ratio_interval(ours, base, CI_DRAWS, CI_SEED)


def stored(per: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {c: {m: {k: v for k, v in s.items() if k != "windows"} for m, s in r.items()} for c, r in per.items()}


def save_report(ws: Path, report: dict[str, Any]) -> None:
    from ..db import get_or_add, session
    from ..db.models import WorkspaceReport

    with session(ws) as s:
        get_or_add(s, WorkspaceReport, kind=REPORT + report["domain"]).data = report


def with_published(report: dict[str, Any]) -> dict[str, Any]:
    from .published import scores

    def merged(configs: dict[str, Any]) -> dict[str, Any]:
        return {c: {**r, **scores(c, r[NAIVE]["mase"])} for c, r in configs.items()}

    per = merged(report.get("configs", {}))
    versions = [{**v, "configs": merged(v["configs"])} if v.get("configs") else v for v in report.get("versions", [])]
    return {**report, "configs": per, "overall": overall(per), "versions": versions} if per else report


def promoted_board(report: dict[str, Any]) -> dict[str, Any]:
    version = promoted_version(report)
    configs = (version or {}).get("configs")
    if not configs:
        return {}
    board = overall(configs)
    for name, s in board.items():
        shared = {c: r for c, r in configs.items() if name in r and version["recipe"] in r}
        if s["datasets"] < len(configs) and shared:
            s["ours_on_shared"] = overall(shared)[version["recipe"]]["mase"]
    return {"configs": configs, "overall": board}


def reports(ws: Path) -> list[dict[str, Any]]:
    from sqlalchemy import select

    from ..db import session
    from ..db.models import WorkspaceReport

    with session(ws) as s:
        rows = s.scalars(select(WorkspaceReport.data).where(WorkspaceReport.kind.startswith(REPORT))).all()
    found = {r["domain"]: with_published(r) for r in rows if r}
    return [found.get(d) or {"domain": d, "at": None} for d in [*DOMAINS, *(k for k in found if k not in DOMAINS)]]
