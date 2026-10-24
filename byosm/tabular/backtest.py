from __future__ import annotations

from collections.abc import Callable
from datetime import date
from typing import Any

from .snapshots import LABEL, Recipe, snapshot

TRAIN_YEARS = 3
TRAIN_STEP = 2


def check_windows(cutoffs: dict[str, list[int]], backtest: list[int], horizon: int) -> None:
    last_train = max(cutoffs["train"]) + horizon
    if last_train > min(cutoffs["test"]):
        raise ValueError(f"train labels reach {last_train}, after the first test cutoff {min(cutoffs['test'])}")
    unknown = [y for y in [*cutoffs["test"], *backtest] if y + horizon > date.today().year]
    if unknown:
        raise ValueError(f"the outcome of test years {unknown} is not known yet with a {horizon} year horizon")


def windows(test: int, horizon: int) -> dict[str, list[int]]:
    last = test - horizon
    return {"train": [last - (TRAIN_YEARS - 1 - i) * TRAIN_STEP for i in range(TRAIN_YEARS)], "test": [test]}


def scorable(splits: dict[str, Any]) -> bool:
    return all(len(v) and v[LABEL].nunique() == 2 for v in splits.values())


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
        if not scorable(splits):
            log(f"  backtest {year}: skipped, train or test lacks ended and alive entities")
            continue
        scores = evaluate(splits, categorical)
        log(f"  backtest {year}: " + ", ".join(f"{m} {s['auc']:.3f}" for m, s in scores.items()))
        out.append({"year": year, "positive": float(splits["test"][LABEL].mean()), "scores": scores})
    return out
