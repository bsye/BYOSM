from __future__ import annotations

import numpy as np

from .recipe import Recipe

AMPLITUDE_KNOTS = 4
AMPLITUDE_DEPTH = 0.5
CENSOR_QUANTILE = 0.3
CENSOR_FROM_BELOW = 0.5
MIX_CONCENTRATION = 1.5


def _seen(segment: np.ndarray, context: int) -> np.ndarray:
    past = segment[:context]
    return past[~np.isnan(past)]


def amplitude(segment: np.ndarray, context: int, rng: np.random.Generator) -> np.ndarray:
    seen = _seen(segment, context)
    level = float(seen.mean()) if len(seen) else 0.0
    knots = np.linspace(0, len(segment) - 1, AMPLITUDE_KNOTS)
    gains = rng.uniform(1 - AMPLITUDE_DEPTH, 1 + AMPLITUDE_DEPTH, AMPLITUDE_KNOTS)
    return level + (segment - level) * np.interp(np.arange(len(segment)), knots, gains)


def censor(segment: np.ndarray, context: int, rng: np.random.Generator) -> np.ndarray:
    seen = _seen(segment, context)
    if not len(seen):
        return segment
    share = rng.uniform(0, CENSOR_QUANTILE)
    if rng.random() < CENSOR_FROM_BELOW:
        return np.maximum(segment, np.quantile(seen, share))
    return np.minimum(segment, np.quantile(seen, 1 - share))


def _unit(segment: np.ndarray, context: int) -> np.ndarray:
    seen = np.abs(_seen(segment, context))
    return segment / (float(seen.mean()) if len(seen) and seen.mean() > 0 else 1.0)


def mix(a: np.ndarray, b: np.ndarray, context: int, rng: np.random.Generator) -> np.ndarray:
    weight = rng.beta(MIX_CONCENTRATION, MIX_CONCENTRATION)
    return weight * _unit(a, context) + (1 - weight) * _unit(b, context)


def augmented(segments: list[np.ndarray], context: int, recipe: Recipe, rng: np.random.Generator) -> list[np.ndarray]:
    out = []
    for segment in segments:
        if recipe.mixup and rng.random() < recipe.mixup:
            segment = mix(segment, segments[int(rng.integers(len(segments)))], context, rng)
        if recipe.amplitude and rng.random() < recipe.amplitude:
            segment = amplitude(segment, context, rng)
        if recipe.censor and rng.random() < recipe.censor:
            segment = censor(segment, context, rng)
        out.append(segment)
    return out


def hidden(past: np.ndarray, patch: int, rate: float, rng: np.random.Generator) -> np.ndarray:
    if not rate:
        return past
    patches = len(past) // patch
    span = int(rng.integers(0, round(2 * rate * patches) + 1))
    start = int(rng.integers(0, patches - span + 1))
    out = past.copy()
    out[start * patch : (start + span) * patch] = np.nan
    return past if np.isnan(out).all() else out
