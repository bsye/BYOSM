from __future__ import annotations

from collections.abc import Callable
from functools import cached_property

import numpy as np

from ..spec import TaskSpec
from .layouts import option_text, params, required
from .pairs import Pair
from .relabel import relabel
from .rewrite import left_out
from .situations import NO_FIT, Situation

SIMILARITY_CHUNK = 256
STOP_WORDS = "english"

Log = Callable[[str], None]
Train = list[tuple[str, Situation]]


def names(s: Situation) -> set[str]:
    return {o["name"] for o in s.options}


def query(s: Situation) -> str:
    return " ".join([s.text, *(option_text(o) for o in s.options if o["name"] in s.chosen)])


class ToolIndex:
    def __init__(self, train: Train):
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.positives = [(g, s) for g, s in train if s.chosen]
        tools: dict[str, dict] = {}
        self.neighbours: dict[str, set[str]] = {}
        for _, s in train:
            for o in s.options:
                tools.setdefault(o["name"], o)
                self.neighbours.setdefault(o["name"], set()).update(names(s) - {o["name"]})
        self.tools, self.index = list(tools.values()), {n: i for i, n in enumerate(tools)}
        self.vectorizer = TfidfVectorizer(stop_words=STOP_WORDS).fit([option_text(t) for t in self.tools])
        self.matrix = self.vectorizer.transform([option_text(t) for t in self.tools])

    @cached_property
    def rows(self) -> list[np.ndarray]:
        out: list[np.ndarray] = []
        for start in range(0, len(self.positives), SIMILARITY_CHUNK):
            part = [query(s) for _, s in self.positives[start : start + SIMILARITY_CHUNK]]
            out += list((self.vectorizer.transform(part) @ self.matrix.T).toarray())
        return out

    def positions(self, found: set[str]) -> list[int]:
        return [self.index[n] for n in found if n in self.index]

    def best(self, row: np.ndarray, excluded: set[str], within: set[str] | None) -> dict | None:
        allowed = np.full(len(self.tools), within is None)
        allowed[self.positions(within or set())] = True
        allowed[self.positions(excluded)] = False
        scores = np.where(allowed, row, -1.0)
        j = int(np.argmax(scores))
        return self.tools[j] if allowed[j] and (within is not None or scores[j] > 0) else None


def _near(index: ToolIndex, within: Callable[[Situation], set[str] | None], kind: str) -> list[Pair]:
    out = []
    for (g, s), row in zip(index.positives, index.rows, strict=True):
        if tool := index.best(row, names(s), within(s)):
            out.append(Pair(g, s.text, tool, [*s.options, tool], NO_FIT, kind=kind))
    return out


def nearest(index: ToolIndex, cfg: dict | None, log: Log) -> list[Pair]:
    return _near(index, lambda s: None, "nearest")


def same_category(index: ToolIndex, cfg: dict | None, log: Log) -> list[Pair]:
    def related(s: Situation) -> set[str]:
        return {n for c in s.chosen for n in index.neighbours.get(c, set())}

    return _near(index, related, "same_category")


def missing_parameter(index: ToolIndex, cfg: dict | None, log: Log) -> list[Pair]:
    jobs = []
    for g, s in index.positives:
        if tool := next((o for o in s.options if o["name"] in s.chosen and required(o)), None):
            jobs.append((g, s, tool, required(tool)[0]))
    asks = [(s.text, f"{p}: {params(tool)[p].get('description') or p}") for _, s, tool, p in jobs]
    texts = left_out(asks, cfg, log)
    return [
        Pair(g, text, tool, s.options, NO_FIT, kind="missing_parameter")
        for (g, s, tool, _), text in zip(jobs, texts, strict=True)
        if text and text != s.text
    ]


def removed(index: ToolIndex, cfg: dict | None, log: Log) -> list[Pair]:
    out = []
    for (g, s), row in zip(index.positives, index.rows, strict=True):
        rest = [o for o in s.options if o["name"] not in s.chosen]
        if tool := index.best(row, s.chosen, {o["name"] for o in rest}):
            out.append(Pair(g, s.text, tool, rest, NO_FIT, kind="removed"))
    return out


MINERS = {
    "nearest": nearest,
    "same_category": same_category,
    "missing_parameter": missing_parameter,
    "removed": removed,
}


def near_misses(spec: TaskSpec, train: Train, cfg: dict | None, log: Log) -> list[Pair]:
    kinds = spec.data.pairs.near_misses if spec.data.pairs else []
    if not kinds or not any(s.chosen for _, s in train):
        return []
    index = ToolIndex(train)
    mined = [p for kind in kinds for p in MINERS[kind](index, cfg, log)]
    return relabel(spec, mined, cfg, log)
