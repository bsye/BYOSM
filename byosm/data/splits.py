from __future__ import annotations

import hashlib
import random
from collections.abc import Callable

from .store import CALIB, DEV_SHARE, TEST_SHARE, near_groups

Log = Callable[[str], None]


def _split3(items: list, rng: random.Random, dev: float = DEV_SHARE, test: float = TEST_SHARE) -> dict[str, list]:
    items = items[:]
    rng.shuffle(items)
    n = len(items)
    a, b = int(n * (1 - dev - test)), int(n * (1 - test))
    return {"train": items[:a], "dev": items[a:b], "test": items[b:]}


def _group_split3(items: list[tuple], rng: random.Random) -> dict[str, list]:
    groups: dict[int, list] = {}
    for g, item in zip(near_groups([x for x, _ in items]), items, strict=True):
        groups.setdefault(g, []).append(item)
    return {s: [item for group in part for item in group] for s, part in _split3(list(groups.values()), rng).items()}


MIN_HOLD_OUT = 50
HELD = ("dev", "test")


def _hold_out(splits: dict[str, list[dict]], rng: random.Random, log: Log) -> None:
    empty = [s for s in HELD if not splits[s]]
    if not empty or len(splits["train"]) < MIN_HOLD_OUT:
        return
    by_label: dict[str, list[dict]] = {}
    for r in splits["train"]:
        label = r.get("label")
        by_label.setdefault(label if isinstance(label, str) else "", []).append(r)
    parts: dict[str, list[dict]] = {"train": [], "dev": [], "test": []}
    for rows in by_label.values():
        for s, part in _group_split3([(r["input"], r) for r in rows], rng).items():
            parts[s] += [r for _, r in part]
    splits["train"] = parts["train"] + [r for s in HELD if s not in empty for r in parts[s]]
    for s in empty:
        splits[s] = parts[s]
    log(
        f"  no source gave {' or '.join(empty)} rows: held out {', '.join(f'{len(parts[s])} for {s}' for s in empty)} from train"
    )


CALIB_EVERY = 2


def _label_key(r: dict) -> str:
    label = r.get("label")
    return label if isinstance(label, str) else ""


def _stable(rid: object) -> str:
    return hashlib.sha1(str(rid).encode()).hexdigest()


def _group_key(r: dict, near: int) -> object:
    meta = r.get("meta") or {}
    return meta.get("file") or meta.get("group") or near


def carve_calib(splits: dict[str, list[dict]]) -> None:
    if splits.get(CALIB):
        return
    groups: dict[object, list[dict]] = {}
    for g, r in zip(near_groups([r["input"] for r in splits["dev"]]), splits["dev"], strict=True):
        groups.setdefault(_group_key(r, g), []).append(r)
    by_label: dict[str, list[list[dict]]] = {}
    for group in groups.values():
        by_label.setdefault(_label_key(group[0]), []).append(group)
    ordered = [g for lab in sorted(by_label) for g in sorted(by_label[lab], key=lambda g: _stable(g[0]["id"]))]
    splits["dev"] = [r for i, g in enumerate(ordered) if not i % CALIB_EVERY for r in g]
    splits[CALIB] = [r for i, g in enumerate(ordered) if i % CALIB_EVERY for r in g]
