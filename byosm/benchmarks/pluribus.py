from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from ..config import STATE_DIR, find_workspace
from ..data.store import record
from ..evals.grouped import CATEGORY
from ..games.poker.pluribus import decisions
from ..games.poker.spots import Spot
from ..options.heldout import hashed
from ..spec import TaskSpec
from . import pokerbench
from .protocol import Kind, Published
from .source import sparse_clone, split_of

REPO = "uoftcprg/phh-dataset"
URL = f"https://github.com/{REPO}.git"
COMMIT = "e47fbd5816372360bade4de5d712346fe1bb70f6"
FOLDER = "data/pluribus"
DECISIONS = 15169
TRAIN_SHARE = 0.5
TRAIN_KEY = "train:"
PUBLISHED = {
    "open-source Pluribus blueprint, exact action in its own bet menu": {
        "all 15,169 decisions": 0.668,
        "preflop": 0.756,
    },
}


def hands() -> list[Path]:
    root = sparse_clone(URL, COMMIT, FOLDER, find_workspace() / STATE_DIR / "datasets" / "phh-dataset")
    return sorted(root.rglob("*.phh"))


def trains_on(hand: str) -> bool:
    return hashed(TRAIN_KEY + hand) < TRAIN_SHARE


def replayed() -> list[tuple[str, int, Spot]]:
    found = []
    for path in hands():
        hand = f"{path.parent.name}/{path.stem}"
        found += [(hand, i, s) for i, s in enumerate(decisions(path.read_text()))]
    if len(found) != DECISIONS:
        raise ValueError(f"the replay found {len(found)} Pluribus decisions, not the {DECISIONS} published")
    return found


def train_spots() -> list[Spot]:
    return [s for hand, _, s in replayed() if trains_on(hand)]


def rows(spec: TaskSpec, log: Callable[[str], None]) -> list[dict]:
    source = f"github:{REPO}"
    out = []
    for hand, i, s in replayed():
        if not trains_on(hand):
            meta = {CATEGORY: s.street, "hand": hand}
            out.append({**record(s.text, s.label, source, **meta), "id": f"{hand}:{i}", "split": split_of(hand)})
    log(f"  pluribus: {len(out)} of {DECISIONS} decisions held out, the rest are for training")
    return out


def published(spec: TaskSpec) -> Published:
    return PUBLISHED


DECISIONS_KIND = Kind(pokerbench.fits, pokerbench.expected, pokerbench.observed, published)
