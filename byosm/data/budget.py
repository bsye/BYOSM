from __future__ import annotations

import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ..llm import get_llm, parallel_calls
from ..spec import TaskSpec
from . import synth_text
from .store import SPLITS

MEASURE_CALLS = 3
MIN_TRAIN_PER_LABEL = 20
MIN_DEV_ROWS = 50
MIN_TEST_ROWS = 100
FIT_FILE = "fit.json"

Log = Callable[[str], None]
Sizes = dict[str, int]


def planned(spec: TaskSpec) -> Sizes:
    return {"train": spec.data.train_size, "dev": spec.data.dev_size, "test": spec.data.test_size}


def floors(spec: TaskSpec) -> Sizes:
    return {"train": MIN_TRAIN_PER_LABEL * len(spec.labels), "dev": MIN_DEV_ROWS, "test": MIN_TEST_ROWS}


def shrink(need: Sizes, have: Sizes, scale: float, floor: Sizes) -> Sizes:
    return {s: min(need[s], max(round(need[s] * scale), floor[s] - have[s])) for s in need}


def _and(items: list[str]) -> str:
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def fit_line(seconds_per_call: float, written: Sizes, plan: Sizes, seconds: float) -> str:
    cut = [s for s in SPLITS if written[s] < plan[s]]
    rows = _and([f"{written[s]} {s}" for s in cut])
    was = _and([str(plan[s]) for s in cut])
    return (
        f"the LLM takes {seconds_per_call:.0f} s per call: writing {rows} rows instead of {was} "
        f"to fit the {seconds / 3600:.1f} h data budget"
    )


def _write(spec: TaskSpec, splits: dict[str, list[dict]], split: str, n: int, cfg: dict | None, log: Log) -> None:
    prior = splits["train"] if split == "train" else splits["train"] + splits["dev"] + splits["test"]
    splits[split] += synth_text.generate(spec, n, split, cfg, against=prior, log=log)


def _seconds_per_call(spec: TaskSpec, split: str, cfg: dict | None, elapsed: float, n: int) -> float:
    made = synth_text.calls(n, spec.data.per_call)
    lanes = get_llm(synth_text.role(split), cfg).parallel(parallel_calls(cfg))
    return elapsed / made * min(made, lanes) / lanes


def _remaining_calls(spec: TaskSpec, need: Sizes) -> int:
    return sum(synth_text.calls(n, spec.data.per_call) for n in need.values() if n)


def add_synthetic(
    spec: TaskSpec, splits: dict[str, list[dict]], cfg: dict | None, seconds: float, log: Log
) -> dict[str, Any]:
    start = time.monotonic()
    plan = planned(spec)
    have = {s: len(splits[s]) for s in SPLITS}
    need = {s: max(0, plan[s] - have[s]) for s in SPLITS}
    fit: dict[str, Any] = {"planned": plan, "budget_hours": round(seconds / 3600, 2), "seconds_per_call": None}
    first = next((s for s in SPLITS if need[s]), None)
    if first:
        probe = min(need[first], MEASURE_CALLS * spec.data.per_call)
        _write(spec, splits, first, probe, cfg, log)
        pace = _seconds_per_call(spec, first, cfg, time.monotonic() - start, probe)
        have[first], need[first] = have[first] + probe, need[first] - probe
        left = max(0.0, seconds - (time.monotonic() - start))
        estimate = pace * _remaining_calls(spec, need)
        if estimate > left:
            need = shrink(need, have, left / estimate, floors(spec))
        fit["seconds_per_call"] = round(pace, 2)
    written = {s: have[s] + need[s] for s in SPLITS}
    fit |= {"written": written, "shrunk": any(written[s] < plan[s] for s in SPLITS)}
    if fit["shrunk"]:
        log(f"  {fit_line(fit['seconds_per_call'], written, plan, seconds)}")
    for s in SPLITS:
        if need[s]:
            _write(spec, splits, s, need[s], cfg, log)
    return fit


def record_fit(data_dir: Path, fit: dict[str, Any], splits: dict[str, list[dict]]) -> None:
    actual = {s: len(splits[s]) for s in SPLITS}
    (data_dir / FIT_FILE).write_text(json.dumps({**fit, "actual": actual}, indent=2))


def load_fit(data_dir: Path) -> dict[str, Any] | None:
    f = data_dir / FIT_FILE
    return json.loads(f.read_text()) if f.exists() else None
