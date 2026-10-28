from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator

from .recipe import RECIPES, V1, V2, Recipe
from .specialist import FIT, MODALITY, PLATEAU, PREDICT, promoted_version

NATIVE = [1]
MULTI_RATES = [1, 2, 4]
PLATEAU_COST = 2.0


class PredictCfg(BaseModel):
    rates: list[int] = Field(default_factory=lambda: list(NATIVE), min_length=1)
    routing: bool = False

    @property
    def multi_rate(self) -> bool:
        return any(r != 1 for r in self.rates)


class FitCfg(BaseModel):
    pretraining: str = V1.name
    quantile_boosting: bool = False

    @field_validator("pretraining")
    @classmethod
    def _known(cls, name: str) -> str:
        if name not in RECIPES:
            raise ValueError(f"no pretraining recipe named {name}: one of {', '.join(RECIPES)}")
        return name

    @property
    def recipe(self) -> Recipe:
        return RECIPES[self.pretraining]


class Options(BaseModel):
    predict: PredictCfg = Field(default_factory=PredictCfg)
    fit: FitCfg = Field(default_factory=FitCfg)


TURNED_ON: dict[str, Any] = {
    f"{PREDICT}.rates": MULTI_RATES,
    f"{PREDICT}.routing": True,
    f"{FIT}.pretraining": V2.name,
    f"{FIT}.quantile_boosting": True,
}


def of(version: dict[str, Any] | None) -> Options:
    found = version or {}
    return Options(predict=found.get(PREDICT) or {}, fit=found.get(FIT) or {})


def fields(technique: Any) -> list[str]:
    from ..improve.paths import matches

    return [f for f in TURNED_ON if any(matches(p, f) for p in technique.paths)]


def forecaster_techniques() -> list[Any]:
    from ..improve.techniques import techniques

    found = [t for t in techniques() if t.fits(MODALITY) and fields(t)]
    return sorted(found, key=lambda t: -t.prior / t.cost)


def turned_on(options: Options, technique: Any) -> Options:
    data = options.model_dump()
    for path in fields(technique):
        section, _, name = path.partition(".")
        data[section][name] = TURNED_ON[path]
    return Options(**data)


def _ready(technique: Any, report: dict[str, Any]) -> bool:
    return technique.cost <= PLATEAU_COST or bool(report.get(PLATEAU))


def next_technique(report: dict[str, Any]) -> tuple[str, Options] | None:
    best = promoted_version(report)
    if best is None:
        return None
    options = of(best)
    tried = [of(v) for v in report.get("versions", []) if v.get("effort") == best.get("effort")]
    for t in forecaster_techniques():
        candidate = turned_on(options, t)
        if _ready(t, report) and candidate != options and candidate not in tried:
            return t.id, candidate
    return None
