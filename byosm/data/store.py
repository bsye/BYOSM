from __future__ import annotations

import json
import uuid
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from ..text import normalize

SPLITS = ("train", "dev", "test")


def read(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def append(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def record(input: Any, label: Any, source: str, **meta: Any) -> dict:
    return {"id": uuid.uuid4().hex[:12], "input": input, "label": label, "source": source, "meta": meta}


def shingles(text: str, n: int = 5) -> set[str]:
    w = normalize(text).split()
    return {" ".join(w[i : i + n]) for i in range(max(1, len(w) - n + 1))}


def dedupe(rows: list[dict], against: list[dict] | None = None, near: float = 0.7) -> list[dict]:
    seen = {normalize(r["input"]) for r in (against or []) if isinstance(r["input"], str)}
    ref = [shingles(r["input"]) for r in (against or []) if isinstance(r["input"], str)]
    index: dict[str, list[int]] = {}
    for i, s in enumerate(ref):
        for g in s:
            index.setdefault(g, []).append(i)
    out = []
    for r in rows:
        if not isinstance(r["input"], str):
            out.append(r)
            continue
        k = normalize(r["input"])
        if not k or k in seen:
            continue
        if ref:
            s = shingles(r["input"])
            hits = Counter(i for g in s for i in index.get(g, ()))
            if any(c / len(s | ref[i]) > near for i, c in hits.most_common(3)):
                continue
        seen.add(k)
        out.append(r)
    return out


def label_counts(rows: list[dict]) -> dict[str, int]:
    c: Counter = Counter()
    for r in rows:
        lab = r["label"]
        if isinstance(lab, list):
            c.update(str(x) for x in lab)
        else:
            c[str(lab)] += 1
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

    def summary(self) -> dict[str, Any]:
        return {s: {"n": len(rows := self.load(s)), "labels": label_counts(rows)} for s in SPLITS}
