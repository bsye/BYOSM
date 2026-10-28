from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..data.store import CALIB, record
from ..evals.grouped import CATEGORY, GROUP
from ..hub.facts import HUB_SOURCE
from ..options.heldout import hashed
from ..options.pairs import pairs_of, rendered, split_layouts
from ..options.situations import Situation
from ..spec import PairPlan, TaskSpec

CALIB_SHARE = 0.2
TEST = "test"


@dataclass
class Case:
    group: str
    category: str
    situation: Situation
    split_key: str = ""


def split_of(key: str) -> str:
    return CALIB if hashed(key) < CALIB_SHARE else TEST


def download(repo: str, file: str) -> Path:
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(repo, file, repo_type="dataset"))


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
