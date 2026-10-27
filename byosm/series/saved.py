from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from .gifteval import DOMAINS, Series, horizon, load, season, splits
from .metrics import QUANTILES

FOLDER = "forecasters"
SUFFIX = ".joblib"
SAMPLE_SEED = 0
HISTORY = 1024


def path(ws: Path, domain: str, version: str) -> Path:
    from ..config import STATE_DIR

    return ws / STATE_DIR / FOLDER / domain / f"{version}{SUFFIX}"


def save(ws: Path, domain: str, version: str, model: Any) -> None:
    import joblib

    out = path(ws, domain, version)
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out)


def saved(ws: Path, domain: str, version: str | None) -> bool:
    return bool(version) and path(ws, domain, version).exists()


def live_version(ws: Path, domain: str, report: dict[str, Any]) -> str | None:
    names = [report.get("promoted"), *(v["name"] for v in reversed(report.get("versions", [])))]
    return next((n for n in names if saved(ws, domain, n)), None)


def _json(values: np.ndarray) -> list[float | None]:
    return [None if np.isnan(v) else float(v) for v in values]


def forecast(ws: Path, domain: str, version: str, history: list[float | None], freq: str) -> dict[str, Any]:
    import joblib

    model = joblib.load(path(ws, domain, version))
    values = np.asarray([np.nan if v is None else v for v in history], dtype=float)
    h = horizon(freq)
    quantiles = model.predict_many([values], season(freq), h)[0]
    return {"horizon": h, "quantiles": list(QUANTILES), "forecast": np.round(quantiles, 4).tolist()}


def example(domain: str, config: str, index: int) -> dict[str, Any]:
    if config not in DOMAINS.get(domain, []):
        raise KeyError(f"{config} is not a dataset of {domain}")
    windows = splits(_series(config))
    w = windows[np.random.default_rng(SAMPLE_SEED + index).integers(len(windows))]
    history, future = _json(w["history"][-HISTORY:]), _json(w["future"])
    return {"config": config, "freq": w["freq"], "item": w["item"], "history": history, "future": future}


@lru_cache(maxsize=len(DOMAINS) * 4)
def _series(config: str) -> list[Series]:
    return load(config)
