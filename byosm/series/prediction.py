from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from .specialist import MODALITY, PREDICT, promoted_version

NATIVE = [1]
MULTI_RATES = [1, 2, 4]


class PredictCfg(BaseModel):
    rates: list[int] = Field(default_factory=lambda: list(NATIVE), min_length=1)
    routing: bool = False

    @property
    def multi_rate(self) -> bool:
        return any(r != 1 for r in self.rates)


TURNED_ON: dict[str, Any] = {"rates": MULTI_RATES, "routing": True}


def of(version: dict[str, Any] | None) -> PredictCfg:
    return PredictCfg(**((version or {}).get(PREDICT) or {}))


def fields(technique: Any) -> list[str]:
    from ..improve.paths import matches

    return [f for f in PredictCfg.model_fields if any(matches(p, f"{PREDICT}.{f}") for p in technique.paths)]


def forecaster_techniques() -> list[Any]:
    from ..improve.techniques import techniques

    found = [t for t in techniques() if t.fits(MODALITY) and fields(t)]
    return sorted(found, key=lambda t: -t.prior / t.cost)


def turned_on(cfg: PredictCfg, technique: Any) -> PredictCfg:
    return PredictCfg(**{**cfg.model_dump(), **{f: TURNED_ON[f] for f in fields(technique)}})


def next_technique(report: dict[str, Any]) -> tuple[str, PredictCfg] | None:
    best = promoted_version(report)
    if best is None:
        return None
    cfg = of(best)
    tried = [of(v) for v in report.get("versions", []) if v.get("effort") == best.get("effort")]
    for t in forecaster_techniques():
        candidate = turned_on(cfg, t)
        if candidate != cfg and candidate not in tried:
            return t.id, candidate
    return None
