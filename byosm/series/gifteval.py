from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

REPO = "Salesforce/GiftEval"
ARROW = "data-00000-of-00001.arrow"
PREDICTION_LENGTH = {"M": 12, "W": 8, "D": 30, "H": 48, "T": 48, "S": 60, "Q": 8, "A": 6, "Y": 6}
SEASON = {"M": 12, "W": 1, "D": 7, "H": 24, "T": 144, "S": 360, "Q": 4, "A": 1, "Y": 1}
MAX_WINDOWS = 20
TEST_SHARE = 0.1
DOMAINS = {
    "energy": ["electricity/H", "solar/H", "ett1/H", "ett2/H"],
    "sales": ["hierarchical_sales/D", "restaurant", "car_parts_with_missing"],
    "operations": ["bizitobs_application", "bizitobs_service", "bizitobs_l2c/H"],
}


@dataclass
class Series:
    item: str
    freq: str
    values: np.ndarray


def unit(freq: str) -> str:
    return next((c for c in freq.upper().lstrip("0123456789") if c.isalpha()), "D")


def load(config: str) -> list[Series]:
    from datasets import Dataset
    from huggingface_hub import hf_hub_download

    ds = Dataset.from_file(hf_hub_download(REPO, f"{config}/{ARROW}", repo_type="dataset"))
    out = []
    for row in ds:
        target = np.asarray(row["target"], dtype=float)
        rows = target if target.ndim == 2 else target[None]
        out += [Series(f"{row['item_id']}:{i}", row["freq"], r) for i, r in enumerate(rows)]
    return out


def horizon(freq: str) -> int:
    return PREDICTION_LENGTH.get(unit(freq), PREDICTION_LENGTH["D"])


def season(freq: str) -> int:
    return SEASON.get(unit(freq), 1)


def windows(series: list[Series]) -> int:
    h, shortest = horizon(series[0].freq), min(len(s.values) for s in series)
    return max(1, min(MAX_WINDOWS, math.ceil(TEST_SHARE * shortest / h)))


def splits(series: list[Series], back: int = 0) -> list[dict[str, Any]]:
    h, w = horizon(series[0].freq), windows(series)
    out = []
    for s in series:
        for k in range(w):
            end = len(s.values) - ((back + 1) * w - k) * h
            if end > 2 * season(s.freq):
                out.append(
                    {"item": s.item, "freq": s.freq, "history": s.values[:end], "future": s.values[end : end + h]}
                )
    return out


def all_configs() -> list[str]:
    from huggingface_hub import list_repo_files

    files = list_repo_files(REPO, repo_type="dataset")
    return sorted({f.rpartition("/")[0] for f in files if f.endswith(ARROW)})
