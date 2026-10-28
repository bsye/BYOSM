from __future__ import annotations

import copy
from typing import Any

from ..spec import TaskSpec
from ..spec.sections import HUB_PREFIX

BACKBONE_PATHS = ("backbone", "document.text_backbone", "document.image_backbone")


def _segment_matches(pattern: str, part: str) -> bool:
    if pattern == "*":
        return part.isdecimal()
    return part in pattern.split("|")


def matches(pattern: str, path: str) -> bool:
    segments, parts = pattern.split("."), path.split(".")
    if len(segments) != len(parts):
        return False
    return all(_segment_matches(s, p) for s, p in zip(segments, parts, strict=True))


def allowed(path: str) -> bool:
    from .techniques import allowed_paths

    return any(matches(a, path) for a in allowed_paths())


def _apply_op(d: dict, o: dict) -> None:
    *head, last = o["path"].split(".")
    cur: Any = d
    for k in head:
        cur = cur[int(k)] if isinstance(cur, list) else cur[k]
    key: Any = int(last) if isinstance(cur, list) else last
    if o["op"] == "append":
        (cur[key] if isinstance(cur, list) else cur.setdefault(key, [])).append(o["value"])
    else:
        cur[key] = o["value"]


def apply_ops(spec: TaskSpec, ops: list[dict]) -> TaskSpec:
    d = copy.deepcopy(spec.model_dump(mode="json"))
    for o in ops:
        _apply_op(d, o)
    return TaskSpec.model_validate(d)


def untrainable(ops: list[dict]) -> list[str]:
    from ..modalities.base import trainable

    repos = {o["value"] for o in ops if o["path"] in BACKBONE_PATHS and str(o["value"]).startswith(HUB_PREFIX)}
    return sorted(r for r in repos if not trainable(r[len(HUB_PREFIX) :]))
