from __future__ import annotations

import json
import random
from dataclasses import dataclass
from typing import Any

FITS, NO_FIT = "fits", "no_fit"
DEFINITIONS = {
    FITS: "the request should call this option now: it does what the request asks, and the request gives every "
    "argument the option requires",
    NO_FIT: "the request should not call this option now: it does something else, or the request leaves out an "
    "argument the option requires, so the assistant should ask for it first",
}
NONE_SHARE = 0.15
TEST_SHARE = 0.15
DEV_SHARE = 0.1


@dataclass
class Situation:
    text: str
    options: list[dict[str, Any]]
    chosen: set[str]


def xlam(row: dict[str, Any]) -> Situation:
    tools = json.loads(row["tools"]) if isinstance(row["tools"], str) else row["tools"]
    answers = json.loads(row["answers"]) if isinstance(row["answers"], str) else row["answers"]
    return Situation(row["query"].strip(), tools, {a["name"] for a in answers})


def with_unrelated(situations: list[Situation], rng: random.Random) -> list[Situation]:
    out = []
    for s in situations:
        if rng.random() < NONE_SHARE:
            other = rng.choice(situations)
            names = {o["name"] for o in s.options}
            if not names & {o["name"] for o in other.options}:
                out.append(Situation(s.text, other.options, set()))
                continue
        out.append(s)
    return out
