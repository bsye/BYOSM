from __future__ import annotations

import random
from collections import Counter, defaultdict
from collections.abc import Callable
from pathlib import Path

from ...benchmarks.pokerbench import REPO, test_spots, train_spots
from ...config import STATE_DIR
from ...data.table import LABEL_COL, SPLIT_COL, write_csv
from ...spec import Label, TaskSpec
from ...spec.sections import DataPlan, Objective
from ...text import normalize
from .moves import DEFINITIONS, LABELS, family
from .spots import Spot

FOLDER = "poker"
TRAIN_SPOTS = 60000
HELD_SHARE = 0.05
TARGET = 0.8
LATENCY_MS = 100.0
EPOCHS = 3
SUITES = ("pokerbench", "pluribus")
POSITIVE = "fold"
Log = Callable[[str], None]


def table_path(ws: Path, name: str) -> Path:
    return ws / STATE_DIR / FOLDER / f"{name}.csv"


def balanced(spots: list[Spot], held: list[Spot], size: int) -> list[Spot]:
    seen = {normalize(s.text) for s in held}
    by_label: dict[str, list[Spot]] = defaultdict(list)
    for s in spots:
        if normalize(s.text) not in seen:
            by_label[s.label].append(s)
    rng = random.Random(FOLDER)
    picked: list[Spot] = []
    share = size // len(LABELS)
    for label in LABELS:
        found = by_label[label]
        rng.shuffle(found)
        picked += found[:share]
    rng.shuffle(picked)
    return picked


def split_at(i: int, held: int) -> str:
    return "dev" if i < held else "test" if i < 2 * held else "train"


def build_table(ws: Path, name: str, size: int, log: Log) -> Path:
    picked = balanced(train_spots(), test_spots(), size)
    held = int(len(picked) * HELD_SHARE)
    path = table_path(ws, name)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_csv(
        path, ["text", LABEL_COL, SPLIT_COL], [[s.text, s.label, split_at(i, held)] for i, s in enumerate(picked)]
    )
    log(f"  wrote {len(picked)} spots to {path}: {dict(Counter(s.label for s in picked))}")
    return path


def spec_for(name: str, description: str, table: Path) -> TaskSpec:
    from ...hub.facts import HUB_SOURCE
    from ...spec.backbones import DEFAULT_BACKBONE
    from ...spec.compiler import train_cfg

    backbone = DEFAULT_BACKBONE["text"]
    labels = [Label(name=n, definition=DEFINITIONS[n], family=family(n)) for n in LABELS]
    return TaskSpec(
        name=name,
        title="Poker move",
        description=description,
        summary="Pick the solver's move among the ones offered in a 6-max no-limit hold'em spot.",
        modality="text",
        labels=labels,
        positive_label=POSITIVE,
        objective=Objective(metric="accuracy", target=TARGET),
        backbone=backbone,
        latency_budget_ms=LATENCY_MS,
        baselines=["llm"],
        data=DataPlan(synthetic=False, local_dir=str(table), local_source=f"{HUB_SOURCE}{REPO}"),
        train=train_cfg("text", backbone).model_copy(update={"epochs": EPOCHS}),
    )


def create(ws: Path, name: str, description: str, log: Log) -> TaskSpec:
    from ...benchmarks.add import add
    from ...benchmarks.catalog import BENCHMARKS
    from ...project import Project

    p = Project(name, ws)
    if p.exists():
        raise ValueError(f"{name} already exists")
    p.ensure()
    p.save_spec(spec_for(name, description, build_table(ws, name, TRAIN_SPOTS, log)))
    for suite in SUITES:
        add(p, BENCHMARKS[suite], log)
    spec = p.spec()
    for e in spec.evals:
        e.primary = e.name == SUITES[0]
    spec.selection_suites = list(SUITES)
    p.save_spec(spec)
    log(f"saved {name}: PokerBench and agreement with Pluribus decide promotion")
    return spec
