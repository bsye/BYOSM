from __future__ import annotations

import csv
from typing import Any

RESULTS = "https://raw.githubusercontent.com/SalesforceAIResearch/gift-eval/main/results/{model}/all_results.csv"
NAIVE_MODEL = "Seasonal_Naive"
RIVALS = ("Timestral-209M", "Timestral-5M")
TERM = "short"
ALIASES = {"car_parts_with_missing": "car_parts"}
MASE_COLUMN = "eval_metrics/MASE[0.5]"
CRPS_COLUMN = "eval_metrics/mean_weighted_sum_quantile_loss"
PROTOCOL_TOLERANCE = 0.02


def _table(model: str) -> dict[str, dict[str, str]]:
    from ..opendata.cache import fetch, table

    url = RESULTS.format(model=model)
    fetch(url, log=lambda _: None)
    with table(url).open() as f:
        return {r["dataset"]: r for r in csv.DictReader(f)}


def _key(config: str, known: dict[str, Any]) -> str | None:
    config = ALIASES.get(config, config)
    if "/" in config:
        return f"{config}/{TERM}"
    found = [k for k in known if k.startswith(config + "/") and k.endswith("/" + TERM)]
    return found[0] if len(found) == 1 else None


def scores(config: str, naive_mase: float) -> dict[str, dict[str, Any]]:
    try:
        naives = _table(NAIVE_MODEL)
        key = _key(config, naives)
        naive = naives.get(key or "")
        if not naive or abs(float(naive[MASE_COLUMN]) / naive_mase - 1) > PROTOCOL_TOLERANCE:
            return {}
        rows = {m: _table(m).get(key or "") for m in RIVALS}
    except (OSError, ValueError, KeyError):
        return {}
    return {
        m: {"mase": float(r[MASE_COLUMN]), "crps": float(r[CRPS_COLUMN]), "published": True}
        for m, r in rows.items()
        if r
    }
