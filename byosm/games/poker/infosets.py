from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import numpy as np

from .teacher import run, strategy, work_dir

RECORDS = ".infosets"
META = ".infosets.json"
SUMMARY = "infosets:"
PAD = 255
BEFORE = ("records", "visited", "hands", "skipped")
AFTER = ("width", "hist")
Log = Callable[[str], None]


def parsed(said: list[str]) -> dict[str, int]:
    found = [line for line in said if SUMMARY in line]
    if not found:
        raise RuntimeError("the poker teacher printed no infoset summary")
    words = found[-1].partition(SUMMARY)[2].replace(",", " ").split()
    before = {k: int(words[words.index(k) - 1]) for k in BEFORE}
    return {**before, **{k: int(words[words.index(k) + 1]) for k in AFTER}}


def layout(width: int, hist: int) -> np.dtype:
    return np.dtype(
        [
            ("bucket", "<u2"),
            ("actions", "u1"),
            ("tokens", "u1"),
            ("offered", "u1", (width,)),
            ("hist", "u1", (hist,)),
            ("visits", "<u4"),
            ("probs", "<f4", (width,)),
        ]
    )


def export(ws: Path, hands_dir: Path, name: str, hands: int, log: Log) -> dict[str, int]:
    out = f"{name}{RECORDS}"
    said = run(ws, hands_dir, ["infosets", "--blueprint", strategy(ws, name), "--hands", str(hands), "--out", out], log)
    meta = parsed(said)
    (work_dir(ws) / f"{name}{META}").write_text(json.dumps(meta))
    return meta


def described(ws: Path, name: str) -> dict[str, int]:
    path = work_dir(ws) / f"{name}{META}"
    if not path.exists():
        raise ValueError(f"no infosets exported for {name!r}: run byosm poker teacher-infosets --strategy {name}")
    return json.loads(path.read_text())


def records(ws: Path, name: str) -> np.memmap:
    meta = described(ws, name)
    found = np.memmap(work_dir(ws) / f"{name}{RECORDS}", dtype=layout(meta["width"], meta["hist"]), mode="r")
    if len(found) != meta["records"]:
        raise ValueError(f"{name}{RECORDS} holds {len(found)} records, not the {meta['records']} it was exported with")
    return found
