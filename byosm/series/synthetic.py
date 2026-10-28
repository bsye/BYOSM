from __future__ import annotations

import numpy as np

PERIODS = {
    "5T": (288.0, 2016.0),
    "15T": (96.0, 672.0),
    "H": (24.0, 168.0),
    "D": (7.0, 365.25),
    "W": (52.18,),
    "M": (12.0,),
    "Q": (4.0,),
}
MIN_LENGTH = 128
SEASON_CHANCE = 0.7
HARMONICS = 3
TREND_SCALE = 3.0
AR_MAX = 0.95
AR_KERNEL = 256
NOISE_RANGE = (0.05, 0.5)
WALK_CHANCE = 0.3
WALK_RANGE = (0.01, 0.2)
SHIFTS_PER_SERIES = 1.0
LEVEL_RANGE = (-5.0, 5.0)
SCALE_RANGE = (0.1, 100.0)
FLOOR_CHANCE = 0.2


def _seasonal(rng: np.random.Generator, t: np.ndarray, periods: tuple[float, ...]) -> np.ndarray:
    out = np.zeros(len(t))
    order = np.arange(1, HARMONICS + 1)
    for period in periods:
        if rng.random() < SEASON_CHANCE:
            weights, phases = rng.normal(0, 1, HARMONICS) / order, rng.uniform(0, 2 * np.pi, HARMONICS)
            out += np.sin(2 * np.pi * np.outer(t, order) / period + phases) @ weights
    return out


def _trend(rng: np.random.Generator, t: np.ndarray) -> np.ndarray:
    x = t / len(t)
    return rng.normal(0, TREND_SCALE) * x + rng.normal(0, TREND_SCALE / 2) * x**2


def _ar(rng: np.random.Generator, n: int) -> np.ndarray:
    phi, shocks = rng.uniform(0, AR_MAX), rng.normal(0, rng.uniform(*NOISE_RANGE), n)
    return np.convolve(shocks, phi ** np.arange(AR_KERNEL))[:n]


def _walk(rng: np.random.Generator, n: int) -> np.ndarray:
    return np.cumsum(rng.normal(0, rng.uniform(*WALK_RANGE), n)) if rng.random() < WALK_CHANCE else np.zeros(n)


def _shifts(rng: np.random.Generator, n: int) -> np.ndarray:
    count = rng.poisson(SHIFTS_PER_SERIES)
    return np.cumsum(np.bincount(rng.integers(0, n, count), weights=rng.normal(0, 1, count), minlength=n))


def series(rng: np.random.Generator, n: int) -> np.ndarray:
    t = np.arange(n, dtype=float)
    periods = list(PERIODS.values())[int(rng.integers(len(PERIODS)))]
    shape = _seasonal(rng, t, periods) + _trend(rng, t) + _ar(rng, n) + _walk(rng, n) + _shifts(rng, n)
    out = np.exp(rng.uniform(*np.log(SCALE_RANGE))) * (rng.uniform(*LEVEL_RANGE) + shape)
    return np.maximum(out, 0.0) if rng.random() < FLOOR_CHANCE else out


class Synthetic:
    def __init__(self, length: int) -> None:
        self.length = length

    def draw(self, rng: np.random.Generator) -> np.ndarray:
        return series(rng, int(rng.integers(MIN_LENGTH, max(MIN_LENGTH, self.length) + 1)))
