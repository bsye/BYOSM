from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from ..config import STATE_DIR, find_workspace
from ..data.store import record
from ..evals.grouped import CATEGORY
from ..games.poker.pluribus import decisions
from ..spec import TaskSpec
from . import pokerbench
from .protocol import Kind, Published, Values
from .source import sparse_clone, split_of

REPO = "uoftcprg/phh-dataset"
URL = f"https://github.com/{REPO}.git"
COMMIT = "e47fbd5816372360bade4de5d712346fe1bb70f6"
FOLDER = "data/pluribus"
DECISIONS = 15169
COUNT = "decisions"
PUBLISHED = {
    "open-source Pluribus blueprint, exact action in its own bet menu": {
        "all 15,169 decisions": 0.668,
        "preflop": 0.756,
    },
}


def hands() -> list[Path]:
    root = sparse_clone(URL, COMMIT, FOLDER, find_workspace() / STATE_DIR / "datasets" / "phh-dataset")
    return sorted(root.rglob("*.phh"))


def rows(spec: TaskSpec, log: Callable[[str], None]) -> list[dict]:
    source = f"github:{REPO}"
    out = []
    for path in hands():
        hand = f"{path.parent.name}/{path.stem}"
        for i, s in enumerate(decisions(path.read_text())):
            meta = {CATEGORY: s.street, "hand": hand}
            out.append({**record(s.text, s.label, source, **meta), "id": f"{hand}:{i}", "split": split_of(hand)})
    return out


def expected(spec: TaskSpec, converted: list[dict]) -> Values:
    return pokerbench.expected(spec, converted) | {COUNT: DECISIONS}


def observed(result: dict[str, Any]) -> Values:
    return pokerbench.observed(result) | {COUNT: result["n_test"] + result["n_calib"]}


def published(spec: TaskSpec) -> Published:
    return PUBLISHED


DECISIONS_KIND = Kind(pokerbench.fits, expected, observed, published)
