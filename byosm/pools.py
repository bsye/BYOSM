from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import TYPE_CHECKING

from .benchmarks import when2call
from .benchmarks.pairs import Case, fits

if TYPE_CHECKING:
    from .spec import TaskSpec

POOLS: dict[str, tuple[Callable[[TaskSpec], bool], Callable[[], Iterator[Case]], str]] = {
    "when2call-train": (fits, when2call.train_cases, when2call.REPO),
}


def pools_for(spec: TaskSpec) -> list[str]:
    return [name for name, (applies, _, _) in POOLS.items() if applies(spec)]
