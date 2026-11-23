from __future__ import annotations

from collections.abc import Callable
from typing import Any

from ..data.store import record
from ..evals.grouped import CATEGORY
from ..games.poker.moves import LABELS, family
from ..games.poker.spots import Raw, Spot, read_raw, render
from ..hub.facts import HUB_SOURCE
from ..spec import TaskSpec
from .protocol import Kind, Published, Values
from .source import TEST, download, split_of

REPO = "RZ412/PokerBench"
FILE = "{street}_{size}_{split}_set_game_scenario_information.csv"
TEST_FILES = {"preflop": "1k", "postflop": "10k"}
TRAIN_FILES = {"preflop": "60k", "postflop": "500k"}
ALL_SPOTS = "all 11k spots"
PUBLISHED = {
    "GPT-4, few-shot": (53.55, 65.54),
    "Llama-3 70B, few-shot": (39.16, 49.78),
    "Llama-3 8B, fine-tuned": (78.26, 80.64),
}
ACCURACY, FAMILY_ACCURACY = "accuracy", "family_accuracy"


def raws(files: dict[str, str], split: str) -> list[Raw]:
    out: list[Raw] = []
    for street, size in files.items():
        out += read_raw(download(REPO, FILE.format(street=street, size=size, split=split)), street == "preflop")
    return out


def test_raws() -> list[Raw]:
    return raws(TEST_FILES, "test")


def train_raws() -> list[Raw]:
    return raws(TRAIN_FILES, "train")


def test_spots() -> list[Spot]:
    return [render(r) for r in test_raws()]


def rows(spec: TaskSpec, log: Callable[[str], None]) -> list[dict]:
    source = HUB_SOURCE + REPO
    out = []
    for i, s in enumerate(test_spots()):
        out.append(
            {
                **record(s.text, s.label, source, **{CATEGORY: s.street}),
                "id": f"pokerbench:{i}",
                "split": split_of(f"{REPO}:{i}"),
            }
        )
    return out


def fits(spec: TaskSpec) -> bool:
    return spec.label_names == LABELS and spec.label_families == [family(n) for n in LABELS]


def expected(spec: TaskSpec, converted: list[dict]) -> Values:
    test = [r for r in converted if r["split"] == TEST]
    positive = spec.label_names[spec.pos_index]
    share = sum(r["label"] == positive for r in test) / len(test)
    same_family = sum(family(r["label"]) == family(positive) for r in test) / len(test)
    return {ACCURACY: share, FAMILY_ACCURACY: same_family}


def observed(result: dict[str, Any]) -> Values:
    return {k: result["metrics"].get(k) for k in (ACCURACY, FAMILY_ACCURACY)}


def published(spec: TaskSpec) -> Published:
    return {
        m: {f"exact move, {ALL_SPOTS}": em / 100, f"action, {ALL_SPOTS}": aa / 100} for m, (em, aa) in PUBLISHED.items()
    }


SPOTS = Kind(fits, expected, observed, published)
