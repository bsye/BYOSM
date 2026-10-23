from __future__ import annotations

from dataclasses import dataclass
from typing import Any

CUTOFF, LABEL, AGE = "_cutoff", "_label", "age_years"
DAYS_PER_YEAR = 365.25


@dataclass
class Recipe:
    start: str
    event: str
    horizon_years: int
    cutoffs: dict[str, list[int]]
    features: list[str]
    per_cutoff: int = 50_000
    seed: int = 0


def _dates(df: Any, col: str) -> Any:
    import pandas as pd

    return pd.to_datetime(df[col], errors="coerce")


def snapshot(df: Any, r: Recipe, year: int) -> Any:
    import pandas as pd

    cut = pd.Timestamp(f"{year}-01-01")
    start, event = _dates(df, r.start), _dates(df, r.event)
    alive = (start < cut) & (event.isna() | (event >= cut))
    rows = df.loc[alive, r.features].copy()
    rows[AGE] = (cut - start[alive]).dt.days / DAYS_PER_YEAR
    rows[LABEL] = (event[alive] < cut + pd.DateOffset(years=r.horizon_years)).astype(int)
    rows[CUTOFF] = year
    return rows.sample(min(len(rows), r.per_cutoff), random_state=r.seed + year)


def build(df: Any, r: Recipe) -> dict[str, Any]:
    import pandas as pd

    return {
        split: pd.concat([snapshot(df, r, y) for y in years], ignore_index=True) for split, years in r.cutoffs.items()
    }
