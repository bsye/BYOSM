from __future__ import annotations

import itertools
from collections.abc import Callable
from pathlib import Path

from ..config import STATE_DIR
from ..spec import Label, TaskSpec
from ..spec.sections import DataPlan, Objective
from .pairs import FITS, NO_FIT, write, xlam

SOURCES = {"lockon/xlam-function-calling-60k": xlam}
LAYA = "hf:convaiinnovations/laya"
FOLDER = "options"
EPOCHS = 3
BATCH = 16
LATENCY_MS = 100.0
TARGET = 0.9


def table_path(ws: Path, name: str) -> Path:
    return ws / STATE_DIR / FOLDER / f"{name}.csv"


def build_table(ws: Path, name: str, source: str, situations: int, log: Callable[[str], None]) -> Path:
    from datasets import load_dataset

    read = SOURCES[source]
    rows = itertools.islice(load_dataset(source, split="train", streaming=True), situations)
    path = table_path(ws, name)
    counts = write(path, [read(r) for r in rows])
    log(f"  wrote {sum(counts.values())} pairs to {path}: {counts}")
    return path


def spec_for(name: str, table: Path, backbone: str = LAYA) -> TaskSpec:
    from ..spec.compiler import train_cfg

    train = train_cfg("text", backbone).model_copy(update={"epochs": EPOCHS, "batch_size": BATCH})
    labels = [
        Label(name=FITS, definition="this option is one the request should use"),
        Label(name=NO_FIT, definition="this option should not be used for the request"),
    ]
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
        data=DataPlan(synthetic=False, local_dir=str(table)),
        train=train,
    )
