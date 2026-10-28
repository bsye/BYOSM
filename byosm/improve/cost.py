from __future__ import annotations

from ..spec import TaskSpec
from ..spec.sections import EMBEDDING_HEAD, SETFIT
from .paths import apply_ops

INGEST_S = {"document": 0.36, "image": 0.02, "audio": 0.05, "text": 0.0}
METHOD_COST = {EMBEDDING_HEAD: 0.05, SETFIT: 0.3}


def _items(spec: TaskSpec) -> int:
    n = 0
    for s in spec.data.hf_sources:
        if s.role == "labeled":
            n += s.per_label * len(spec.labels) if s.sampling == "balanced" and s.format == "rows" else s.max_rows
    return n + (spec.data.train_size if spec.data.synthetic else 0)


def _encoder_changed(old: TaskSpec, new: TaskSpec) -> bool:
    return (
        new.backbone != old.backbone
        or new.document.image_backbone != old.document.image_backbone
        or new.document.text_backbone != old.document.text_backbone
    )


def _ingest_per_item(spec: TaskSpec) -> float:
    ocr = spec.modality == "document" and spec.document.route != "visual"
    if ocr:
        return INGEST_S["document"]
    if spec.modality == "document":
        return INGEST_S["image"]
    return INGEST_S.get(spec.modality, 0)


def estimate(dg: dict, pr: dict) -> dict[str, float]:
    old = dg["spec"]
    try:
        new = apply_ops(old, pr["ops"])
    except Exception:
        return {"minutes": float("nan"), "new_items": 0}
    a, b = max(1, _items(old)), _items(new)
    scale = (b / a) * (new.train.epochs / max(1, old.train.epochs))
    if _encoder_changed(old, new):
        scale *= 1.5
    scale *= METHOD_COST.get(new.train.method, 1.0) / METHOD_COST.get(old.train.method, 1.0)
    ingest = max(0, b - a) * _ingest_per_item(new)
    return {"minutes": round(((dg.get("train_seconds") or 0) * scale + ingest) / 60, 1), "new_items": max(0, b - a)}
