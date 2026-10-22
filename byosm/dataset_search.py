from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .benchmark_search import PICKER_ROLE, RULE, prefer_real_rows, try_sources
from .config import load_config
from .data import hf_source
from .data.hf_source.mapping import pick_sources
from .data.hf_source.research import research
from .llm import role_model
from .project import Project
from .spec import HFSource, TaskSpec

MAX_EXTRA_SOURCES = 3
STEP = "datasets"
SUGGESTIONS = "data-suggestions"
UNMAPPED = "picked, but its columns or labels do not map onto ours"


def find_datasets(spec: TaskSpec, cfg: dict | None, log: Callable[[str], None]) -> dict[str, Any]:
    have = {s.dataset for s in spec.data.hf_sources} | {e.source.dataset for e in spec.evals if e.source}
    queries = [*spec.data.hf_queries, *spec.search.get("benchmark", []), *([spec.title] if spec.title else [])]
    found = [*research(spec, cfg or load_config(), log), *hf_source.search(queries)]
    hits = [h for h in {h["id"]: h for h in reversed(found)}.values() if h["id"] not in have][::-1]
    if not hits:
        return {"mapped": [], "why": {}, "unmapped": [], "by": RULE, "searched": False}
    picks = pick_sources(spec, hits, cfg)[:MAX_EXTRA_SOURCES]
    tmp = spec.model_copy(deep=True)
    tmp.data.hf_sources = [src for src, _ in picks]
    mapped = hf_source.resolve(tmp, cfg, log)
    added = {m.dataset for m in mapped}
    return {
        "mapped": mapped,
        "why": {s.dataset: w for s, w in picks},
        "unmapped": [s for s, _ in picks if s.dataset not in added],
        "by": role_model(PICKER_ROLE, cfg),
        "searched": True,
    }


def _add_datasets(spec: TaskSpec, cfg: dict | None, log: Callable[[str], None], decisions: list[dict]) -> None:
    found = find_datasets(spec, cfg, log)
    spec.data.hf_sources.extend(found["mapped"])
    added = sorted(m.dataset for m in found["mapped"])
    why = "; ".join(f"{d}: {found['why'].get(d, '')}" for d in added)
    if not found["searched"]:
        why = "no other dataset on the Hub matches the searches"
    alts = [
        {
            "option": s.dataset,
            "why_not": UNMAPPED,
            "args": try_sources(spec.name, [{"dataset": s.dataset, "role": s.role}], s.dataset.split("/")[-1]),
        }
        for s in found["unmapped"]
    ]
    decisions.append(
        {
            "step": STEP,
            "choice": ", ".join(added) or None,
            "why": why or "none of the search results fit the task",
            "by": found["by"],
            "alternatives": alts,
        }
    )
    if any(m.role == "labeled" for m in found["mapped"]):
        prefer_real_rows(spec, decisions)


def add_datasets(spec: TaskSpec, cfg: dict | None, log: Callable[[str], None], decisions: list[dict]) -> None:
    try:
        _add_datasets(spec, cfg, log, decisions)
    except Exception as e:
        why = f"dataset search failed: {type(e).__name__}: {e}"
        decisions.append({"step": STEP, "choice": None, "why": why, "by": RULE, "alternatives": []})


def _with(spec: TaskSpec, src: HFSource) -> list[dict]:
    return [s.model_dump(mode="json", exclude_defaults=True) for s in [*spec.data.hf_sources, src]]


def suggest_datasets(p: Project, spec: TaskSpec, cfg: dict | None, log: Callable[[str], None]) -> list[dict]:
    found = find_datasets(spec, cfg, log)
    out = [
        {
            "dataset": m.dataset,
            "role": m.role,
            "why": found["why"].get(m.dataset, ""),
            "by": found["by"],
            "args": try_sources(spec.name, _with(spec, m), m.dataset.split("/")[-1]),
        }
        for m in found["mapped"]
    ]
    p.save_report(SUGGESTIONS, out)
    log(f"{len(out)} datasets fit {spec.name}" + "".join(f"\n  - {s['dataset']}: {s['why']}" for s in out))
    return out


def saved_suggestions(p: Project) -> list[dict] | None:
    return p.report(SUGGESTIONS)
