from __future__ import annotations

from dataclasses import dataclass
from typing import Any

WINDOW_YEARS = 1
NEVER_DAYS = 9999


@dataclass
class History:
    name: str
    events: Any
    key: str
    date: str


def features(h: History, keys: Any, cut: Any) -> Any:
    import pandas as pd

    when = pd.to_datetime(h.events[h.date], errors="coerce")
    before = h.events.assign(_when=when)[when < cut]
    recent = before[before["_when"] >= cut - pd.DateOffset(years=WINDOW_YEARS)]
    count = recent.groupby(h.key).size()
    last = before.groupby(h.key)["_when"].max()
    out = pd.DataFrame(index=keys.index)
    out[f"{h.name}_last_year"] = count.reindex(keys.to_numpy()).fillna(0).astype(int).to_numpy()
    out[f"{h.name}_days_since"] = (cut - last.reindex(keys.to_numpy())).dt.days.fillna(NEVER_DAYS).to_numpy()
    return out
