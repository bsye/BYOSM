from __future__ import annotations

import time
from collections.abc import Callable
from datetime import date
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, model_validator
from sqlalchemy import delete, select

from ..config import STATE_DIR
from ..db import get_or_add, session
from ..db.models import WorkspaceReport
from .backtest import windows
from .history import History
from .snapshots import Recipe

PREFIX = "forecast:"
FOLDER = "forecasts"


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


def create(ws: Path, name: str, data: dict[str, Any]) -> ForecastSpec:
    spec = ForecastSpec.model_validate(data)
    _check_windows(spec)
    _write(ws, name, {"name": name, "spec": spec.model_dump(), "result": None})
    return spec


def load(ws: Path, name: str) -> ForecastSpec:
    row = _row(ws, name)
    if not row:
        raise KeyError(f"no forecast named {name}")
    return ForecastSpec.model_validate(row["spec"])


def _check_windows(spec: ForecastSpec) -> None:
    last_train = max(spec.cutoffs["train"]) + spec.horizon_years
    if last_train > min(spec.cutoffs["test"]):
        raise ValueError(f"train labels reach {last_train}, after the first test cutoff {min(spec.cutoffs['test'])}")
    unknown = [y for y in [*spec.cutoffs["test"], *spec.backtest] if y + spec.horizon_years > date.today().year]
    if unknown:
        raise ValueError(
            f"the outcome of test years {unknown} is not known yet with a {spec.horizon_years} year horizon"
        )


def _read(spec: ForecastSpec, log: Callable[[str], None]) -> Any:
    import pandas as pd

    from ..opendata.cache import fetch, table
    from .derive import derive

    def read(url: str, file: str | None = None, columns: list[str] | None = None) -> Any:
        fetch(url, log)
        path = table(url, file)
        log(f"  reading {path.name}")
        return pd.read_csv(path, low_memory=False, dtype=str, usecols=columns, encoding="utf-8-sig")

    histories = [
        History(name, pd.concat([read(u, columns=[h.key, h.date]) for u in h.sources]), h.key, h.date)
        for name, h in spec.histories.items()
    ]
    return derive(read(spec.source, spec.file), spec.derived), histories


def run(ws: Path, name: str, log: Callable[[str], None] = print) -> dict[str, Any]:
    from .fit import evaluate
    from .leakage import check
    from .snapshots import LABEL, build

    spec = load(ws, name)
    _check_windows(spec)
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
    now, everyone = today(df, recipe, kept, list(dropped), spec.shown or [spec.key or spec.start], log)
    scored_path(ws, name).parent.mkdir(parents=True, exist_ok=True)
    everyone.to_csv(scored_path(ws, name), index=False)
    result = {
        "at": time.time(),
        "splits": sizes,
        "leakage": leaks,
        "dropped": dropped,
        "scores": scores,
        "over_time": over_time,
        "today": now,
    }
    _write(ws, name, {"name": name, "spec": spec.model_dump(), "result": result})
    return result


def remove(ws: Path, name: str) -> None:
    if not _row(ws, name):
        raise KeyError(f"no forecast named {name}")
    with session(ws) as s:
        s.execute(delete(WorkspaceReport).where(WorkspaceReport.kind == PREFIX + name))
    scored_path(ws, name).unlink(missing_ok=True)


def scored_path(ws: Path, name: str) -> Path:
    return ws / STATE_DIR / FOLDER / f"{name}.csv"


def forecasts(ws: Path) -> list[dict[str, Any]]:
    with session(ws) as s:
        rows = s.scalars(select(WorkspaceReport.data).where(WorkspaceReport.kind.startswith(PREFIX))).all()
    return sorted((r for r in rows if r), key=lambda r: r["name"])
