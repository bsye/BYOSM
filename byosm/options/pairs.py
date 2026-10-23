from __future__ import annotations

import csv
import json
import random
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

FITS, NO_FIT = "fits", "no_fit"
NONE_SHARE = 0.15
TEST_SHARE = 0.15
DEV_SHARE = 0.1
DESCRIPTION_CHARS = 300
OPTION_MARK = "\n\nOption: "
OTHERS_MARK = "\nOther options: "
NO_OTHERS = "none"


@dataclass
class Situation:
    text: str
    options: list[dict[str, Any]]
    chosen: set[str]


def option_text(option: dict[str, Any]) -> str:
    params = ", ".join((option.get("parameters") or {}).keys())
    return f"{option['name']}: {str(option.get('description') or '')[:DESCRIPTION_CHARS]} ({params})"


def pair_text(situation: str, option: dict[str, Any], others: list[dict[str, Any]] | None = None) -> str:
    names = ", ".join(o["name"] for o in others or [] if o["name"] != option["name"]) or NO_OTHERS
    return f"{situation}{OPTION_MARK}{option_text(option)}{OTHERS_MARK}{names}"


def xlam(row: dict[str, Any]) -> Situation:
    tools = json.loads(row["tools"]) if isinstance(row["tools"], str) else row["tools"]
    answers = json.loads(row["answers"]) if isinstance(row["answers"], str) else row["answers"]
    return Situation(row["query"], tools, {a["name"] for a in answers})


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


def rows(situations: Iterable[Situation], rng: random.Random) -> Iterator[dict[str, str]]:
    for i, s in enumerate(situations):
        draw = rng.random()
        split = "test" if draw < TEST_SHARE else "dev" if draw < TEST_SHARE + DEV_SHARE else "train"
        for o in s.options:
            yield {
                "text": pair_text(s.text, o, s.options),
                "label": FITS if o["name"] in s.chosen else NO_FIT,
                "split": split,
                "group": str(i),
            }


def write(path: Path, situations: list[Situation], seed: int = 0) -> dict[str, int]:
    rng = random.Random(seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    with path.open("w", newline="") as f:
        out = csv.DictWriter(f, fieldnames=["text", "label", "split", "group"])
        out.writeheader()
        for r in rows(with_unrelated(situations, rng), rng):
            out.writerow(r)
            counts[f"{r['split']}:{r['label']}"] = counts.get(f"{r['split']}:{r['label']}", 0) + 1
    return counts
