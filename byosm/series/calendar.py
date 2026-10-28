from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .gifteval import unit

SECONDS = {"S": 1, "T": 60, "H": 3600, "D": 86400, "W": 604800}
MONTHS = {"M": 1, "Q": 3, "A": 12, "Y": 12}
FIELDS = ("minute", "hour", "dayofweek", "day", "month")
DIGITS = "0123456789"
UNKNOWN = -1.0


@dataclass(frozen=True)
class Clock:
    start: np.datetime64
    freq: str

    def stamps(self, n: int) -> np.ndarray:
        digits = self.freq[: len(self.freq) - len(self.freq.lstrip(DIGITS))]
        steps, kind = np.arange(n) * int(digits or 1), unit(self.freq)
        if kind in MONTHS:
            return (self.start.astype("datetime64[M]") + steps * MONTHS[kind]).astype("datetime64[s]")
        return self.start.astype("datetime64[s]") + steps * SECONDS.get(kind, SECONDS["D"])


def clock(start: Any, freq: str) -> Clock | None:
    return None if start is None else Clock(np.datetime64(start, "s"), freq)


def features(at: Clock | None, n: int) -> np.ndarray:
    import pandas as pd

    if at is None:
        return np.full((n, len(FIELDS)), UNKNOWN)
    index = pd.DatetimeIndex(at.stamps(n))
    return np.column_stack([np.asarray(getattr(index, f), dtype=float) for f in FIELDS])
