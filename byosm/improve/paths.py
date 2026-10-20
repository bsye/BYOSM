from __future__ import annotations

import copy
from typing import Any

from ..spec import TaskSpec

ALLOWED = [
    "data.hf_sources",
    "data.hf_sources.*.per_label|max_rows|max_files|partition",
    "data.hf_queries",
    "data.train_size|dev_size|test_size",
    "data.audio.snr_db|fill_ms|max_events|label_by_energy|clip_seconds",
    "data.synthetic|per_call|judge",
    "hard_cases",
    "diversity_axes",
    "backbone",
    "train.epochs|pretrain_steps|pretrain_lambda|lr_backbone|lr_head|batch_size|freeze_backbone|refine_rounds"
    "|refine_examples|pool|layer_decay_floor|probe_epochs|ema|precision|seed",
    "document.route|text_backbone|image_backbone|max_pages|max_len|image_size|ocr|dim|modality_dropout",
]


def _segment_matches(pattern: str, part: str) -> bool:
    if pattern == "*":
        return part.isdecimal()
    return part in pattern.split("|")


def _matches(pattern: str, parts: list[str]) -> bool:
    segments = pattern.split(".")
    if len(segments) != len(parts):
        return False
    return all(_segment_matches(s, p) for s, p in zip(segments, parts, strict=True))


def allowed(path: str) -> bool:
    parts = path.split(".")
    return any(_matches(a, parts) for a in ALLOWED)


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
