from __future__ import annotations

import json
from collections import Counter
from collections.abc import Callable

from ..data.store import STORED_SPLITS, record
from ..spec import PairPlan, TaskSpec
from .pairs import Pair, pairs_of, rendered
from .situations import FITS, Situation

Log = Callable[[str], None]


def situations_of(rows: list[dict[str, str]]) -> list[tuple[str, str, Situation]]:
    groups: dict[str, list[dict[str, str]]] = {}
    for r in rows:
        groups.setdefault(r["group"], []).append(r)
    out = []
    for g, part in groups.items():
        offered = [json.loads(r["option"]) for r in part]
        chosen = {o["name"] for o, r in zip(offered, part, strict=True) if r["label"] == FITS}
        out.append((g, part[0]["split"], Situation(part[0]["request"], offered, chosen)))
    return out


def held_records(rows: list[dict[str, str]], source: str) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {s: [] for s in STORED_SPLITS}
    for r in rows:
        if r["split"] != "train":
            out[r["split"]].append(record(r["text"], r["label"], source, group=r["group"], layout=r["layout"]))
    return out


def train_record(pair: Pair, plan: PairPlan) -> dict:
    text, layout = rendered(pair, plan.layouts, plan.mask_names)
    return record(text, pair.label, pair.source, group=pair.group, layout=layout, kind=pair.kind)


def train_pairs(spec: TaskSpec, train: list[tuple[str, Situation]], cfg: dict | None, log: Log) -> list[Pair]:
    source = spec.data.local_source
    return [p for g, s in train for p in pairs_of(g, s, source)]


def pair_splits(spec: TaskSpec, rows: list[dict[str, str]], cfg: dict | None, log: Log) -> dict[str, list[dict]]:
    plan = spec.data.pairs or PairPlan()
    out = held_records(rows, spec.data.local_source)
    train = [(g, s) for g, split, s in situations_of(rows) if split == "train"]
    pairs = train_pairs(spec, train, cfg, log)
    out["train"] = [train_record(p, plan) for p in pairs]
    kinds = Counter(p.kind for p in pairs)
    log(f"  pairs: {len(train)} training requests, {dict(kinds)} rows in layouts {', '.join(plan.layouts)}")
    return out
