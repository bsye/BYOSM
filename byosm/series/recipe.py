from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .metrics import QUANTILES

QUANTILE_STEP = 0.05
TRAINING_QUANTILES = (0.01, *(round(QUANTILE_STEP * k, 2) for k in range(1, 20)), 0.99)
MASK_RATE = 0.15
AMPLITUDE_CHANCE = 0.3
CENSOR_CHANCE = 0.1
MIXUP_CHANCE = 0.3
SYNTHETIC_SHARE = 0.2


@dataclass(frozen=True)
class Recipe:
    name: str
    tag: str = ""
    quantiles: tuple[float, ...] = QUANTILES
    arcsinh: bool = False
    mask_rate: float = 0.0
    amplitude: float = 0.0
    censor: float = 0.0
    mixup: float = 0.0
    synthetic: float = 0.0
    sizes: tuple[str, ...] = ("base", "large")

    @property
    def served(self) -> list[int]:
        return [self.quantiles.index(q) for q in QUANTILES]

    def scale(self, x: np.ndarray) -> np.ndarray:
        return np.arcsinh(x) if self.arcsinh else x

    def unscale(self, x: np.ndarray) -> np.ndarray:
        return np.sinh(x) if self.arcsinh else x

    def weights(self, size: str) -> str:
        return f"{size}-{self.tag}" if self.tag else size

    def named(self, model: str) -> str:
        return f"{model} {self.tag}" if self.tag else model


V1 = Recipe(name="v1")
V2 = Recipe(
    name="v2",
    tag="v2",
    quantiles=TRAINING_QUANTILES,
    arcsinh=True,
    mask_rate=MASK_RATE,
    amplitude=AMPLITUDE_CHANCE,
    censor=CENSOR_CHANCE,
    mixup=MIXUP_CHANCE,
    synthetic=SYNTHETIC_SHARE,
    sizes=("base", "long"),
)
RECIPES = {r.name: r for r in (V1, V2)}
