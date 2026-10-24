from __future__ import annotations

import time
from typing import Any

from .gifteval import DOMAINS

NAIVE = "seasonal naive"
METRIC = "MASE vs naive"
KIND = "forecast"
RIVAL_NAMES = (NAIVE, "Timestral")
CI_DRAWS = 500
CI_SEED = 0


def is_rival(model: str) -> bool:
    return model.startswith(RIVAL_NAMES)


def choose(overall: dict[str, dict[str, Any]]) -> str | None:
    recipes = {m: s["calib_mase"] for m, s in overall.items() if not is_rival(m) and "calib_mase" in s}
    return min(recipes, key=lambda m: recipes[m]) if recipes else None


def add_version(report: dict[str, Any], interval: list[float] | None, effort: int = 0) -> dict[str, Any]:
    recipe = choose(report["overall"])
    if recipe is None:
        return report
    versions = report.get("versions", [])
    scores = report["overall"][recipe]
    version = {
        "name": f"v{len(versions) + 1}",
        "recipe": recipe,
        "effort": effort,
        "calib": scores["calib_mase"],
        "test": scores["mase"],
        "crps": scores["crps"],
        "ci": interval,
        "at": time.time(),
    }
    best = next((v for v in versions if v["name"] == report.get("promoted")), None)
    promoted = version["name"] if best is None or version["calib"] < best["calib"] else best["name"]
    return {**report, "versions": [*versions, version], "promoted": promoted}


def best_rival(overall: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
    rivals = {m: s["mase"] for m, s in overall.items() if is_rival(m)}
    if not rivals:
        return None
    name = min(rivals, key=lambda m: rivals[m])
    return {"name": name, "score": rivals[name]}


def summary(report: dict[str, Any]) -> dict[str, Any]:
    domain = report["domain"]
    versions = report.get("versions", [])
    promoted = next((v for v in versions if v["name"] == report.get("promoted")), None)
    return {
        "name": domain,
        "kind": KIND,
        "title": f"{domain.capitalize()} forecasting",
        "created": report.get("created"),
        "description": " · ".join(DOMAINS.get(domain, [domain])),
        "modality": "series",
        "task_type": KIND,
        "labels": 0,
        "metric": METRIC,
        "higher_is_better": False,
        "primary_suite": "GIFT-Eval",
        "benchmark": "GIFT-Eval",
        "promoted": report.get("promoted"),
        "score": promoted["test"] if promoted else None,
        "rival": best_rival(report.get("overall") or {}),
        "experiments": [v["name"] for v in versions],
        "exported": False,
        "part_of": None,
    }
