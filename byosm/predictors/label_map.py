from __future__ import annotations

import random
from collections import Counter
from typing import Any

import torch

from ..spec import TaskSpec
from .hf import BATCH, load_model

SUGGEST_ROWS_PER_LABEL = 30
SUGGEST_SEED = 0


def _sample(rows: list[dict], spec: TaskSpec) -> list[dict]:
    rng = random.Random(SUGGEST_SEED)
    by: dict[str, list[dict]] = {n: [] for n in spec.label_names}
    for r in rows:
        if isinstance(r.get("label"), str) and r["label"] in by:
            by[r["label"]].append(r)
    return [r for group in by.values() for r in rng.sample(group, min(SUGGEST_ROWS_PER_LABEL, len(group)))]


@torch.no_grad()
def _predict(proc: Any, model: Any, texts: list[str]) -> list[int]:
    out = []
    for i in range(0, len(texts), BATCH):
        enc = proc(texts[i : i + BATCH], padding=True, truncation=True, return_tensors="pt")
        out += model(**enc).logits.argmax(-1).tolist()
    return out


def suggest_label_map(spec: TaskSpec, rows: list[dict], repo: str) -> dict[str, Any]:
    sample = _sample(rows, spec)
    if not sample:
        raise ValueError("no labelled rows to compare against")
    proc, model = load_model(repo, visual=False)
    names = model.config.id2label
    got = _predict(proc, model, [str(r["input"]) for r in sample])
    pairs = Counter((names[k], r["label"]) for k, r in zip(got, sample, strict=True))
    best: dict[str, tuple[str, int]] = {}
    for (theirs, ours), n in pairs.items():
        if n > best.get(theirs, ("", 0))[1]:
            best[theirs] = (ours, n)
    agree = sum(n for _, n in best.values())
    return {
        "map": {t: o for t, (o, _) in best.items()},
        "agreement": agree / len(sample),
        "rows": len(sample),
        "counts": {f"{t}={o}": n for (t, o), n in pairs.items()},
    }
