from __future__ import annotations

import csv
import json
import random
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..spec import PairPlan
from .heldout import EVAL_SPLITS, assign
from .layouts import masked, render
from .situations import FITS, NO_FIT, Situation, with_unrelated

TABLE_COLUMNS = ["text", "label", "split", "group", "request", "option", "layout"]
TABLE_KIND = "table"


@dataclass
class Pair:
    group: str
    request: str
    option: dict[str, Any]
    offered: list[dict[str, Any]]
    label: str
    source: str = ""
    kind: str = TABLE_KIND


def pairs_of(group: str, s: Situation, source: str = "", kind: str = TABLE_KIND) -> list[Pair]:
    return [
        Pair(group, s.text, o, s.options, FITS if o["name"] in s.chosen else NO_FIT, source, kind) for o in s.options
    ]


def rendered(pair: Pair, layouts: list[str], mask: float) -> tuple[str, str]:
    rng = random.Random(f"{pair.group}|{pair.request}|{pair.option['name']}|{pair.kind}")
    layout = rng.choice(layouts)
    option, offered = masked(pair.option, pair.offered) if rng.random() < mask else (pair.option, pair.offered)
    return render(layout, pair.request, option, offered), layout


def split_layouts(plan: PairPlan, split: str) -> tuple[list[str], float]:
    if split in EVAL_SPLITS:
        return plan.layouts + plan.held_out_layouts, 0.0
    return plan.layouts, plan.mask_names if split == "train" else 0.0


def table_rows(situations: list[Situation], plan: PairPlan, seed: int = 0) -> Iterator[dict[str, str]]:
    rng = random.Random(seed)
    mixed = with_unrelated(situations, rng)
    placed = assign(mixed, iter(rng.random, None), plan.held_out_apis, bool(plan.held_out_layouts))
    for i, (s, split) in enumerate(placed):
        layouts, mask = split_layouts(plan, split)
        for p in pairs_of(str(i), s):
            text, layout = rendered(p, layouts, mask)
            row = {"text": text, "label": p.label, "split": split, "group": p.group, "request": p.request}
            yield row | {"option": json.dumps(p.option), "layout": layout}


def write(path: Path, situations: list[Situation], plan: PairPlan, seed: int = 0) -> dict[str, int]:
    path.parent.mkdir(parents=True, exist_ok=True)
    counts: Counter = Counter()
    with path.open("w", newline="") as f:
        out = csv.DictWriter(f, fieldnames=TABLE_COLUMNS)
        out.writeheader()
        for r in table_rows(situations, plan, seed):
            out.writerow(r)
            counts[f"{r['split']}:{r['label']}"] += 1
    return dict(counts)
