from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import partial

from ..spec import TaskSpec
from . import bfcl, eot, when2call
from .pairs import PAIRS
from .protocol import Kind

Log = Callable[[str], None]


@dataclass(frozen=True)
class Benchmark:
    name: str
    dataset: str
    kind: Kind
    rows: Callable[[TaskSpec, Log], list[dict]]
    about: str


BENCHMARKS = {
    b.name: b
    for b in (
        Benchmark(
            "bfcl",
            bfcl.REPO,
            PAIRS,
            partial(bfcl.rows, bfcl.NON_LIVE),
            "BFCL v3 simple, multiple and irrelevance: one pair per offered function, fitting when the answer calls it",
        ),
        Benchmark(
            "bfcl-live",
            bfcl.REPO,
            PAIRS,
            partial(bfcl.rows, bfcl.LIVE),
            "BFCL v3 live simple, multiple and irrelevance, user-contributed requests and functions",
        ),
        Benchmark(
            "when2call",
            when2call.REPO,
            PAIRS,
            when2call.rows,
            "When2Call multiple-choice test: a tool fits only when the right answer is to call it now",
        ),
        Benchmark(
            "eot-bench",
            eot.REPO,
            eot.TURNS,
            eot.rows,
            "LiveKit eot-bench in the spec's language: the words heard at each moment of each pause, turn over or not",
        ),
    )
}


def fitting(spec: TaskSpec) -> list[Benchmark]:
    text = spec.modality == "text" and spec.task_type == "classify"
    return [b for b in BENCHMARKS.values() if text and b.kind.fits(spec)]
