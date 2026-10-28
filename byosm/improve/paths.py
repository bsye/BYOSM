from __future__ import annotations

import copy
from typing import Any

from ..spec import TaskSpec
from ..spec.backbones import SENTENCE_ENCODERS, embedding_model
from ..spec.sections import FINETUNE, HUB_PREFIX

BACKBONE_PATHS = ("backbone", "document.text_backbone", "document.image_backbone")
VALUE_MARK = "="


def _segment_matches(pattern: str, part: str) -> bool:
    if pattern == "*":
        return part.isdecimal()
    return part in pattern.split("|")


def matches(pattern: str, path: str) -> bool:
    segments, parts = pattern.split("."), path.split(".")
    if len(segments) != len(parts):
        return False
    return all(_segment_matches(s, p) for s, p in zip(segments, parts, strict=True))


def claims(pattern: str, op: dict) -> bool:
    where, mark, value = pattern.partition(VALUE_MARK)
    return matches(where, op["path"]) and (not mark or str(op.get("value")) == value)


def allowed(path: str) -> bool:
    from .techniques import allowed_paths

    return any(matches(a.partition(VALUE_MARK)[0], path) for a in allowed_paths())


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


def _embedding_encoder(old: TaskSpec, new: TaskSpec, ops: list[dict]) -> TaskSpec:
    switched = new.train.method not in (old.train.method, FINETUNE)
    if not switched or new.backbone in SENTENCE_ENCODERS or any(o["path"] == "backbone" for o in ops):
        return new
    return new.model_copy(update={"backbone": embedding_model(new.languages)})


def apply_ops(spec: TaskSpec, ops: list[dict]) -> TaskSpec:
    d = copy.deepcopy(spec.model_dump(mode="json"))
    for o in ops:
        _apply_op(d, o)
    return _embedding_encoder(spec, TaskSpec.model_validate(d), ops)


def untrainable(ops: list[dict]) -> list[str]:
    from ..modalities.base import trainable

    repos = {o["value"] for o in ops if o["path"] in BACKBONE_PATHS and str(o["value"]).startswith(HUB_PREFIX)}
    return sorted(r for r in repos if not trainable(r[len(HUB_PREFIX) :]))
