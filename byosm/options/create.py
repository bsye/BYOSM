from __future__ import annotations

import itertools
from collections.abc import Callable
from pathlib import Path

from ..config import STATE_DIR
from ..spec import Label, PairPlan, TaskSpec
from ..spec.sections import DataPlan, Objective
from .pairs import write
from .situations import FITS, NO_FIT, xlam

SOURCES = {"lockon/xlam-function-calling-60k": xlam}
LAYA = "hf:convaiinnovations/laya"
FOLDER = "options"
EPOCHS = 3
BATCH = 16
LATENCY_MS = 100.0
TARGET = 0.9
HELD_OUT_LAYOUTS = ["prose"]


def table_path(ws: Path, name: str) -> Path:
    return ws / STATE_DIR / FOLDER / f"{name}.csv"


def pair_plan(held_out: bool) -> PairPlan:
    return PairPlan(held_out_apis=True, held_out_layouts=HELD_OUT_LAYOUTS) if held_out else PairPlan()


def build_table(ws: Path, name: str, source: str, situations: int, plan: PairPlan, log: Callable[[str], None]) -> Path:
    from datasets import load_dataset

    read = SOURCES[source]
    rows = itertools.islice(load_dataset(source, split="train", streaming=True), situations)
    path = table_path(ws, name)
    counts = write(path, [read(r) for r in rows], plan)
    log(f"  wrote {sum(counts.values())} pairs to {path}: {counts}")
    return path


def spec_for(name: str, table: Path, source: str, plan: PairPlan, backbone: str = LAYA) -> TaskSpec:
    from ..hub.facts import HUB_SOURCE
    from ..spec.compiler import train_cfg

    train = train_cfg("text", backbone).model_copy(update={"epochs": EPOCHS, "batch_size": BATCH})
    labels = [
        Label(name=FITS, definition="this option is one the request should use"),
        Label(name=NO_FIT, definition="this option should not be used for the request"),
    ]
    data = DataPlan(synthetic=False, local_dir=str(table), local_source=f"{HUB_SOURCE}{source}", pairs=plan)
    return TaskSpec(
        name=name,
        title="Tool choice",
        description="Pick the tools that fit a request from the tools offered with it, or none",
        summary="Decide whether one offered tool should be used for a request; the chosen tools are the ones that fit.",
        modality="text",
        labels=labels,
        positive_label=FITS,
        objective=Objective(metric="f1", target=TARGET),
        backbone=backbone,
        latency_budget_ms=LATENCY_MS,
        data=data,
        train=train,
    )
