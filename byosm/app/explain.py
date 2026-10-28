from __future__ import annotations

import math
from typing import Any

MAX_STEPS = 40
MAX_DRIVERS = 80
DIGITS = 4


def _cuts(n: int) -> list[int]:
    stride = max(1, math.ceil(n / MAX_STEPS))
    return sorted({*range(stride, n + 1, stride), n})


def _p(model: Any, text: str, label: str) -> float:
    return round(float(model.predict(text)["probs"].get(label, 0.0)), DIGITS) if text.strip() else 0.0


def explain(model: Any, text: str, follow: str | None = None) -> dict[str, Any]:
    if follow is not None and follow not in model.labels:
        raise ValueError(f"{follow} is not one of the labels: {', '.join(model.labels)}")
    final = model.predict(text)
    label, words = follow or final["label"], text.split()
    cuts = _cuts(len(words))
    reading = [{"word": words[k - 1], "at": k, "p": _p(model, " ".join(words[:k]), label)} for k in cuts]
    full = final["probs"].get(label, 0.0)
    drivers = [
        {"word": w, "at": i, "drop": round(full - _p(model, " ".join(words[:i] + words[i + 1 :]), label), DIGITS)}
        for i, w in enumerate(words[:MAX_DRIVERS])
    ]
    decision = model.m.get("decision") or {}
    positive = model.labels[model.m["positive_index"]] if model.m.get("positive_index") is not None else None
    threshold = decision.get("threshold") if positive == label else None
    return {
        "label": final["label"],
        "follows": label,
        "words": words,
        "probs": final["probs"],
        "reading": reading,
        "drivers": drivers,
        "threshold": threshold,
    }
