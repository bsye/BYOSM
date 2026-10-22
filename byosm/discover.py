from __future__ import annotations

import json
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from .benchmark_search import add_benchmark
from .data.hf_source.hub import search_terms
from .dataset_search import add_datasets
from .events import stage
from .spec import TaskSpec
from .text import match_label

TASK_TAG = {
    ("text", "classify"): "text-classification",
    ("image", "classify"): "image-classification",
    ("document", "classify"): "image-classification",
    ("audio", "classify"): "audio-classification",
    ("audio", "frame"): "voice-activity-detection",
    ("text", "frame"): "token-classification",
}
LABEL_FETCH_WORKERS = 8
CLASSIFIER_SUFFIX = "ForSequenceClassification"
DEFAULT_NUM_LABELS = 2
MAX_CANDIDATES = 30
SCORABLE_TAGS = {"text-classification", "image-classification", "token-classification"}


def _model_labels(repo: str) -> list[str] | None:
    from huggingface_hub import hf_hub_download

    try:
        cfg = json.loads(Path(hf_hub_download(repo, "config.json")).read_text())
    except Exception:
        return None
    if cfg.get("id2label"):
        return list(cfg["id2label"].values())
    classifier = any(a.endswith(CLASSIFIER_SUFFIX) for a in cfg.get("architectures") or [])
    return [f"LABEL_{i}" for i in range(int(cfg.get("num_labels") or DEFAULT_NUM_LABELS))] if classifier else []


def _search_models(q: str, tag: str) -> list[Any]:
    from huggingface_hub import HfApi

    try:
        return list(HfApi().list_models(search=q, pipeline_tag=tag or None, sort="downloads", limit=15))
    except Exception:
        return []


def model_candidates(spec: TaskSpec, queries: list[str], typed: bool = False) -> list[dict[str, Any]]:
    tag = TASK_TAG.get((spec.modality, spec.task_type))
    if tag not in SCORABLE_TAGS:
        return []
    tag = "" if typed else tag
    terms = queries if typed else search_terms(queries)
    models = {m.id: m for q in terms for m in _search_models(q, tag)}
    top = sorted(models.values(), key=lambda m: -(getattr(m, "downloads", 0) or 0))[:MAX_CANDIDATES]
    with ThreadPoolExecutor(LABEL_FETCH_WORKERS) as pool:
        labels = list(pool.map(_model_labels, [m.id for m in top]))
    n = max(1, len(spec.label_names))
    return [
        {
            "repo": m.id,
            "downloads": getattr(m, "downloads", 0) or 0,
            "coverage": len({match_label(x, spec.label_names) for x in theirs or []} - {None}) / n,
            "labels": list(theirs or []),
            "tag": getattr(m, "pipeline_tag", None) or tag,
        }
        for m, theirs in zip(top, labels, strict=True)
    ]


def baseline_models(spec: TaskSpec, queries: list[str], k: int = 2, min_overlap: float = 0.6) -> list[tuple[str, str]]:
    fits = [c for c in model_candidates(spec, queries) if c["coverage"] >= min_overlap]
    why = "{tag} model, {downloads} downloads, covers {coverage:.0%} of our labels"
    return [(c["repo"], why.format(**c)) for c in fits[:k]]


def _default_baselines(spec: TaskSpec, bl: list[str], decisions: list[dict]) -> None:
    if spec.modality in ("text", "document", "image") and spec.task_type == "classify" and "llm" not in bl:
        bl.append("llm")
        why = "the zero-shot LLM is the System-2 bar the specialist must beat"
        decisions.append({"step": "baseline", "choice": "llm", "why": why})
    if spec.modality == "audio" and spec.task_type == "frame":
        for b in ("energy", "silero"):
            if b not in bl:
                bl.append(b)
        why = "frame-level audio: a sanity floor + LiveKit's VAD"
        decisions.append({"step": "baseline", "choice": "energy, silero", "why": why})


def _model_baselines(spec: TaskSpec, queries: list[str], bl: list[str], decisions: list[dict]) -> None:
    try:
        for repo, why in baseline_models(spec, queries):
            if f"hf:{repo}" not in bl:
                bl.append(f"hf:{repo}")
                decisions.append({"step": "baseline", "choice": f"hf:{repo}", "why": why})
    except Exception as e:
        decisions.append({"step": "baseline", "choice": None, "why": f"model search failed: {type(e).__name__}"})


@stage("Find a benchmark and rivals")
def discover(
    spec: TaskSpec, queries: dict[str, list[str]], cfg: dict | None = None, log: Callable[[str], None] = print
) -> tuple[TaskSpec, list[dict]]:
    spec = spec.model_copy(deep=True)
    decisions: list[dict] = []
    if spec.task_type == "classify" and queries.get("benchmark"):
        add_benchmark(spec, queries["benchmark"], cfg, log, decisions)
    if spec.task_type == "classify":
        add_datasets(spec, cfg, log, decisions)
    bl = list(spec.baselines)
    _default_baselines(spec, bl, decisions)
    _model_baselines(spec, queries.get("baseline") or [], bl, decisions)
    spec.baselines = bl
    return spec, decisions
