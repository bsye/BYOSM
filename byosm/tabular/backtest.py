from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .snapshots import LABEL, Recipe, snapshot

TRAIN_YEARS = 3
TRAIN_STEP = 2


def windows(test: int, horizon: int) -> dict[str, list[int]]:
    last = test - horizon
    return {"train": [last - (TRAIN_YEARS - 1 - i) * TRAIN_STEP for i in range(TRAIN_YEARS)], "test": [test]}


def backtest(
    df: Any, r: Recipe, years: list[int], categorical: list[str], dropped: list[str], log: Callable[[str], None]
) -> list[dict[str, Any]]:
    import pandas as pd

    from .fit import evaluate

    seen: dict[int, Any] = {}

    def cohort(year: int) -> Any:
        if year not in seen:
            seen[year] = snapshot(df, r, year).drop(columns=dropped)
        return seen[year]

    out = []
    for year in sorted(years):
        cut = windows(year, r.horizon_years)
        splits = {k: pd.concat([cohort(y) for y in ys], ignore_index=True) for k, ys in cut.items()}
        scores = evaluate(splits, categorical)
        log(f"  backtest {year}: " + ", ".join(f"{m} {s['auc']:.3f}" for m, s in scores.items()))
        out.append({"year": year, "positive": float(splits["test"][LABEL].mean()), "scores": scores})
    return out
