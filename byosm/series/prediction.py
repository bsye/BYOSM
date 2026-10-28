from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from .specialist import PREDICT

NATIVE = [1]
MULTI_RATES = [1, 2, 4]


class PredictCfg(BaseModel):
    rates: list[int] = Field(default_factory=lambda: list(NATIVE), min_length=1)
    routing: bool = False

    @property
    def multi_rate(self) -> bool:
        return self.rates != NATIVE


def of(version: dict[str, Any] | None) -> PredictCfg:
    return PredictCfg(**((version or {}).get(PREDICT) or {}))
