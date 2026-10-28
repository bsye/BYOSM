from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from ..data.store import record
from ..evals.grouped import CATEGORY, GROUP, PAIR_LABELS, REQUESTS
from ..hub.facts import HUB_SOURCE
from ..options.pairs import pairs_of, rendered, split_layouts
from ..options.situations import FITS, Situation
from ..spec import PairPlan, TaskSpec
from .protocol import Kind, Values
from .source import TEST, split_of


@dataclass
class Case:
    group: str
    category: str
    situation: Situation
    split_key: str = ""


def option(tool: dict[str, Any]) -> dict[str, Any]:
    shape = tool.get("parameters") or {}
    needed = set(shape.get("required") or [])
    props = {k: v if isinstance(v, dict) else {} for k, v in (shape.get("properties") or {}).items()}
    params = {k: {**v, "required": k in needed} for k, v in props.items()}
    return {"name": tool["name"], "description": tool.get("description") or "", "parameters": params}


def pair_rows(spec: TaskSpec, cases: Iterable[Case], dataset: str) -> list[dict]:
    layouts, mask = split_layouts(spec.data.pairs or PairPlan(), TEST)
    source = HUB_SOURCE + dataset
    rows = []
    for c in cases:
        split = split_of(c.split_key or c.group)
        for i, p in enumerate(pairs_of(c.group, c.situation, source)):
            text, layout = rendered(p, layouts, mask)
            meta = {GROUP: c.group, CATEGORY: c.category, "layout": layout}
            rows.append({**record(text, p.label, source, **meta), "id": f"{c.group}:{i}", "split": split})
    return rows


def fits(spec: TaskSpec) -> bool:
    return set(spec.label_names) == PAIR_LABELS and spec.label_names[spec.pos_index] == FITS


def expected(spec: TaskSpec, rows: list[dict]) -> Values:
    test = [r for r in rows if r["split"] == TEST]
    share = sum(r["label"] == FITS for r in test) / max(1, len(test))
    requests: dict[str, list[bool]] = {}
    for r in test:
        requests.setdefault(r["meta"][GROUP], []).append(r["label"] == FITS)
    needed = [all(found) for found in requests.values() if any(found)]
    return {
        "precision": share,
        "recall": 1.0 if share else 0.0,
        "f1": 2 * share / (1 + share),
        "tool_needed": sum(needed) / len(needed) if needed else None,
        "no_tool": 0.0 if len(needed) < len(requests) else None,
    }


def observed(result: dict[str, Any]) -> Values:
    m = result["metrics"]
    found = {k: m[REQUESTS]["all"][k]["accuracy"] for k in ("tool_needed", "no_tool")}
    return {k: m[k] for k in ("precision", "recall", "f1")} | found


PAIRS = Kind(fits, expected, observed)
