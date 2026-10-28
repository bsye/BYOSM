from __future__ import annotations

import hashlib
from collections.abc import Iterator

from ..data.store import CALIB
from .situations import DEV_SHARE, TEST_SHARE, Situation

HELD_API_SHARE = 0.25
CALIB_OF_HELD = 0.5
HASH_CHARS = 8
EVAL_SPLITS = (CALIB, "test")


def held_api(name: str) -> bool:
    digest = int(hashlib.sha1(name.encode()).hexdigest()[:HASH_CHARS], 16)
    return digest / 16**HASH_CHARS < HELD_API_SHARE


def by_api(situation: Situation, draw: float) -> tuple[Situation, str] | None:
    key = situation.chosen or {o["name"] for o in situation.options}
    held = any(held_api(n) for n in key)
    kept = [o for o in situation.options if held_api(o["name"]) == held]
    if not kept:
        return None
    split = (CALIB if draw < CALIB_OF_HELD else "test") if held else ("dev" if draw < DEV_SHARE else "train")
    return Situation(situation.text, kept, situation.chosen & {o["name"] for o in kept}), split


def by_draw(draw: float, marks_calib: bool) -> str:
    if draw < TEST_SHARE:
        return "test"
    if marks_calib and draw < TEST_SHARE + DEV_SHARE * CALIB_OF_HELD:
        return CALIB
    return "dev" if draw < TEST_SHARE + DEV_SHARE else "train"


def assign(
    situations: list[Situation], draws: Iterator[float], apis: bool, marks_calib: bool
) -> list[tuple[Situation, str]]:
    if apis:
        return [found for s in situations if (found := by_api(s, next(draws)))]
    return [(s, by_draw(next(draws), marks_calib)) for s in situations]
