from __future__ import annotations

from collections.abc import Callable

from .benchmark_search import PICKER_ROLE, RULE, prefer_real_rows, try_sources
from .data import hf_source
from .data.hf_source.mapping import pick_sources
from .llm import role_model
from .spec import TaskSpec

MAX_EXTRA_SOURCES = 3
STEP = "datasets"


def _alternative(spec: TaskSpec, dataset: str, role: str, why_not: str) -> dict:
    args = try_sources(spec.name, [{"dataset": dataset, "role": role}], dataset.split("/")[-1])
    return {"option": dataset, "why_not": why_not, "args": args}


def _add_datasets(spec: TaskSpec, cfg: dict | None, log: Callable[[str], None], decisions: list[dict]) -> None:
    have = {s.dataset for s in spec.data.hf_sources} | {e.source.dataset for e in spec.evals if e.source}
    queries = [*spec.data.hf_queries, *spec.search.get("benchmark", [])]
    hits = [h for h in hf_source.search(queries) if h["id"] not in have]
    if not hits:
        decisions.append(
            {
                "step": STEP,
                "choice": None,
                "why": "no other dataset on the Hub matches the searches",
                "by": RULE,
                "alternatives": [],
            }
        )
        return
    by = role_model(PICKER_ROLE, cfg)
    picks = pick_sources(spec, hits, cfg)[:MAX_EXTRA_SOURCES]
    tmp = spec.model_copy(deep=True)
    tmp.data.hf_sources = [src for src, _ in picks]
    mapped = hf_source.resolve(tmp, cfg, log)
    added = {m.dataset for m in mapped}
    spec.data.hf_sources.extend(mapped)
    unmapped = [
        _alternative(spec, s.dataset, s.role, "picked, but its columns or labels do not map onto ours")
        for s, _ in picks
        if s.dataset not in added
    ]
    why = (
        "; ".join(f"{s.dataset}: {w}" for s, w in picks if s.dataset in added)
        or "none of the search results fit the task"
    )
    decisions.append(
        {
            "step": STEP,
            "choice": ", ".join(sorted(added)) or None,
            "why": why,
            "by": by,
            "alternatives": unmapped,
        }
    )
    if any(m.role == "labeled" for m in mapped):
        prefer_real_rows(spec, decisions)


def add_datasets(spec: TaskSpec, cfg: dict | None, log: Callable[[str], None], decisions: list[dict]) -> None:
    try:
        _add_datasets(spec, cfg, log, decisions)
    except Exception as e:
        why = f"dataset search failed: {type(e).__name__}: {e}"
        decisions.append({"step": STEP, "choice": None, "why": why, "by": RULE, "alternatives": []})
