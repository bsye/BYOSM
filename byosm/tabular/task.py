from __future__ import annotations

import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
from pydantic import BaseModel, Field, model_validator
from sqlalchemy import delete, select

from ..config import STATE_DIR
from ..db import get_or_add, session
from ..db.models import WorkspaceReport
from .backtest import check_windows, windows
from .history import History
from .snapshots import Recipe

PREFIX = "forecast:"
FOLDER = "forecasts"
NAME_PUNCTUATION = "-_"


class HistorySpec(BaseModel):
    sources: list[str]
    key: str
    date: str


class ForecastSpec(BaseModel):
    title: str
    source: str
    file: str | None = None
    start: str
    event: str
    horizon_years: int
    cutoffs: dict[str, list[int]] = Field(default_factory=dict)
    test_year: int | None = None
    backtest: list[int] = Field(default_factory=list)
    features: list[str]
    categorical: list[str] = Field(default_factory=list)
    derived: dict[str, dict[str, Any]] = Field(default_factory=dict)
    excluded: dict[str, str] = Field(default_factory=dict)
    per_cutoff: int = 50_000
    key: str | None = None
    shown: list[str] = Field(default_factory=list)
    histories: dict[str, HistorySpec] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _windows(self) -> ForecastSpec:
        if not self.cutoffs and self.test_year:
            self.cutoffs = windows(self.test_year, self.horizon_years)
        if "train" not in self.cutoffs or "test" not in self.cutoffs:
            raise ValueError("give cutoffs with train and test years, or a test_year")
        return self

    def recipe(self, histories: list[History]) -> Recipe:
        fields = (self.start, self.event, self.horizon_years, self.cutoffs, self.features, self.per_cutoff)
        return Recipe(*fields, key=self.key, histories=histories)


def _row(ws: Path, name: str) -> dict[str, Any] | None:
    with session(ws) as s:
        return s.scalar(select(WorkspaceReport.data).where(WorkspaceReport.kind == PREFIX + name))


def _write(ws: Path, name: str, data: dict[str, Any]) -> None:
    with session(ws) as s:
        get_or_add(s, WorkspaceReport, kind=PREFIX + name).data = data


def check_name(name: str) -> str:
    if not name or name.startswith(".") or not all(c.isalnum() or c in NAME_PUNCTUATION for c in name):
        raise ValueError(f"a forecast name uses only letters, digits and {NAME_PUNCTUATION}, not {name!r}")
    return name


def create(ws: Path, name: str, data: dict[str, Any]) -> ForecastSpec:
    check_name(name)
    spec = ForecastSpec.model_validate(data)
    check_windows(spec.cutoffs, spec.backtest, spec.horizon_years)
    _clear(ws, name)
    _write(
        ws, name, {"name": name, "created": datetime.now(UTC).isoformat(), "spec": spec.model_dump(), "result": None}
    )
    return spec


def load(ws: Path, name: str) -> ForecastSpec:
    row = _row(ws, name)
    if not row:
        raise KeyError(f"no forecast named {name}")
    return ForecastSpec.model_validate(row["spec"])


def _read(spec: ForecastSpec, log: Callable[[str], None]) -> Any:
    from .derive import derive

    return derive(read_table(spec.source, log, spec.file), spec.derived), histories(spec, log)


def read_table(url: str, log: Callable[[str], None], file: str | None = None, columns: list[str] | None = None) -> Any:
    import pandas as pd

    from ..opendata.cache import fetch, table

    fetch(url, log)
    path = table(url, file)
    log(f"  reading {path.name}")
    return pd.read_csv(path, low_memory=False, dtype=str, usecols=columns, encoding="utf-8-sig")


def histories(spec: ForecastSpec, log: Callable[[str], None]) -> list[History]:
    import pandas as pd

    return [
        History(name, pd.concat([read_table(u, log, columns=[h.key, h.date]) for u in h.sources]), h.key, h.date)
        for name, h in spec.histories.items()
    ]


def run(ws: Path, name: str, log: Callable[[str], None] = print) -> dict[str, Any]:
    from .fit import evaluate
    from .leakage import check
    from .snapshots import LABEL, build

    spec = load(ws, name)
    check_windows(spec.cutoffs, spec.backtest, spec.horizon_years)
    from .derive import typed

    df, histories = _read(spec, log)
    df, categorical = typed(df, spec.features, spec.categorical)
    splits = build(df, spec.recipe(histories))
    sizes = {k: {"n": len(v), "positive": round(float(v[LABEL].mean()), 4)} for k, v in splits.items()}
    log(f"  snapshots: {sizes}")
    leaks = check(splits["train"], categorical)
    dropped = {x["feature"]: x["why"] for x in leaks if x["suspect"]}
    for feature, why in dropped.items():
        log(f"  left out {feature}: {why}")
    kept = [c for c in categorical if c not in dropped]
    scores = evaluate({k: v.drop(columns=list(dropped)) for k, v in splits.items()}, kept)
    for model, s in scores.items():
        log(f"  {model}: AUC {s['auc']:.3f} ({s['auc_ci'][0]:.3f} to {s['auc_ci'][1]:.3f}), AP {s['ap']:.3f}")
    from .backtest import backtest
    from .score import today

    recipe = spec.recipe(histories)
    over_time = backtest(df, recipe, spec.backtest, kept, list(dropped), log)
    now, everyone, fitted = today(df, recipe, kept, list(dropped), spec.shown or [spec.key or spec.start], log)
    scored_path(ws, name).parent.mkdir(parents=True, exist_ok=True)
    everyone.to_csv(scored_path(ws, name), index=False)
    joblib.dump(fitted, model_path(ws, name))
    result = {
        "at": time.time(),
        "splits": sizes,
        "leakage": leaks,
        "dropped": dropped,
        "scores": scores,
        "over_time": over_time,
        "today": now,
    }
    _write(ws, name, {**(_row(ws, name) or {}), "name": name, "spec": spec.model_dump(), "result": result})
    return result


def remove(ws: Path, name: str) -> None:
    if not _row(ws, name):
        raise KeyError(f"no forecast named {name}")
    with session(ws) as s:
        s.execute(delete(WorkspaceReport).where(WorkspaceReport.kind == PREFIX + name))
    _clear(ws, name)


def _clear(ws: Path, name: str) -> None:
    scored_path(ws, name).unlink(missing_ok=True)
    model_path(ws, name).unlink(missing_ok=True)


def model_path(ws: Path, name: str) -> Path:
    return scored_path(ws, name).with_suffix(".joblib")


def scored_path(ws: Path, name: str) -> Path:
    return ws / STATE_DIR / FOLDER / f"{check_name(name)}.csv"


def forecasts(ws: Path) -> list[dict[str, Any]]:
    with session(ws) as s:
        rows = s.scalars(select(WorkspaceReport.data).where(WorkspaceReport.kind.startswith(PREFIX))).all()
    return sorted((r for r in rows if r), key=lambda r: r["name"])
