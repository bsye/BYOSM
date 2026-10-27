from __future__ import annotations

import json
import uuid
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from ..db.rows import append, exists, read, tally, write
from ..text import normalize

DEV_SHARE = 0.1
TEST_SHARE = 0.15

SPLITS = ("train", "dev", "test")
TRAIN_INPUTS = "train-inputs"
REFINE_PREFIX = "refine-"
ROUNDS_SUFFIX = "-rounds"
BASE_TRAIN = ("train",)


def refine_split(exp: str) -> str:
    return f"{REFINE_PREFIX}{exp}"


def rounds_split(exp: str) -> str:
    return f"{REFINE_PREFIX}{exp}{ROUNDS_SUFFIX}"


NEAR_DUPLICATE = 0.7
NEAR_CANDIDATES = 3


def read_file(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def record(input: Any, label: Any, source: str, **meta: Any) -> dict:
    return {"id": uuid.uuid4().hex[:12], "input": input, "label": label, "source": source, "meta": meta}


def shingles(text: str, n: int = 5) -> set[str]:
    w = normalize(text).split()
    return {" ".join(w[i : i + n]) for i in range(max(1, len(w) - n + 1))}


class ShingleIndex:
    def __init__(self, rows: list[dict]):
        self.rows = [r for r in rows if isinstance(r["input"], str)]
        self.texts = [r["input"] for r in self.rows]
        self.exact = {normalize(t): i for i, t in enumerate(self.texts)}
        self.ref = [shingles(t) for t in self.texts]
        self.index: dict[str, list[int]] = {}
        for i, s in enumerate(self.ref):
            for g in s:
                self.index.setdefault(g, []).append(i)

    def add(self, text: str) -> int:
        i = len(self.texts)
        self.texts.append(text)
        self.exact.setdefault(normalize(text), i)
        self.ref.append(shingles(text))
        for g in self.ref[i]:
            self.index.setdefault(g, []).append(i)
        return i

    def nearest(self, text: str) -> tuple[int, float] | None:
        s = shingles(text)
        hits = Counter(i for g in s for i in self.index.get(g, ()))
        scored = [(i, c / len(s | self.ref[i])) for i, c in hits.most_common(NEAR_CANDIDATES)]
        return max(scored, key=lambda t: t[1], default=None)


def near_groups(texts: list[Any], near: float = NEAR_DUPLICATE) -> list[int]:
    index, of_text, out = ShingleIndex([]), [], []
    for i, t in enumerate(texts):
        if not isinstance(t, str):
            out.append(i)
            continue
        same = index.exact.get(normalize(t))
        hit = index.nearest(t) if same is None else None
        g = of_text[same] if same is not None else of_text[hit[0]] if hit and hit[1] > near else i
        index.add(t)
        of_text.append(g)
        out.append(g)
    return out


def dedupe(rows: list[dict], against: list[dict] | None = None, near: float = NEAR_DUPLICATE) -> list[dict]:
    ref = ShingleIndex(against or [])
    seen = set(ref.exact)
    out = []
    for r in rows:
        if not isinstance(r["input"], str):
            out.append(r)
            continue
        k = normalize(r["input"])
        if not k or k in seen:
            continue
        hit = ref.nearest(r["input"])
        if hit is not None and hit[1] > near:
            continue
        seen.add(k)
        out.append(r)
    return out


def label_counts(rows: list[dict]) -> dict[str, int]:
    return count_labels((r["label"], 1) for r in rows)


def count_labels(tallied: Iterable[tuple[Any, int]]) -> dict[str, int]:
    c: Counter = Counter()
    for lab, n in tallied:
        for x in lab if isinstance(lab, list) else [lab]:
            c[str(x)] += n
    return dict(c)


class Dataset:
    def __init__(self, data_dir: Path):
        self.dir = data_dir

    def path(self, split: str) -> Path:
        return self.dir / f"{split}.jsonl"

    def load(self, split: str) -> list[dict]:
        return read(self.path(split))

    def save(self, split: str, rows: list[dict]) -> None:
        write(self.path(split), rows)

    def add(self, split: str, rows: list[dict]) -> None:
        append(self.path(split), rows)

    def unlabeled(self) -> list[dict]:
        return read(self.dir / "unlabeled.jsonl")

    def built(self) -> bool:
        return exists(self.path("train"))

    def summary(self) -> dict[str, Any]:
        tallies = {s: tally(self.path(s), "label") for s in SPLITS}
        return {s: {"n": sum(n for _, n in t), "labels": count_labels(t)} for s, t in tallies.items()}
