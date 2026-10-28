from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from ..format import fmt
from ..options.situations import FITS, NO_FIT
from ..spec import TaskSpec

GROUP, CATEGORY = "group", "category"
REQUESTS = "requests"
PAIR_LABELS = {FITS, NO_FIT}
SUMMARY_DIGITS = 3


def _requests(rows: list[dict], keep: Callable[[dict], bool]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for r in rows:
        meta = r.get("meta") or {}
        if GROUP in meta and keep(meta):
            out.setdefault(str(meta[GROUP]), []).append(r)
    return out


def per_request(spec: TaskSpec, rows: list[dict], dec: np.ndarray) -> dict[str, Any]:
    from ..options.select import evaluate

    chosen = {r["id"] for r, d in zip(rows, dec, strict=True) if d == spec.pos_index}

    def pick(_: str, part: list[dict]) -> set[int]:
        return {i for i, r in enumerate(part) if r["id"] in chosen}

    categories = sorted({str(c) for r in rows if (c := (r.get("meta") or {}).get(CATEGORY))})
    by = {c: evaluate(_requests(rows, lambda m, c=c: m.get(CATEGORY) == c), pick) for c in categories}
    return {"all": evaluate(_requests(rows, lambda m: True), pick), "by_category": by}


def grouped_metrics(
    spec: TaskSpec, calib: list[dict], test: list[dict], pc: Any, pt: np.ndarray, dec: np.ndarray
) -> dict[str, Any]:
    meta = (test[0].get("meta") or {}) if test and spec.task_type == "classify" else {}
    if GROUP in meta and set(spec.label_names) == PAIR_LABELS:
        return {REQUESTS: per_request(spec, test, dec)}
    return {}


def summary(m: dict[str, Any]) -> str | None:
    if REQUESTS in m:
        s = m[REQUESTS]["all"]
        none, needed = (fmt(s[k]["accuracy"], SUMMARY_DIGITS) for k in ("no_tool", "tool_needed"))
        return f"no tool {none}, tool needed {needed}"
    return None
