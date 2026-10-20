from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .data import hf_source
from .llm import get_llm
from .spec import EvalSuite, HFSource, TaskSpec
from .text import normalize

TASK_TAG = {
    ("text", "classify"): "text-classification",
    ("image", "classify"): "image-classification",
    ("document", "classify"): "image-classification",
    ("audio", "classify"): "audio-classification",
    ("audio", "frame"): "voice-activity-detection",
    ("text", "frame"): "token-classification",
}

HELD_OUT_SPLITS = ("test", "validation", "val", "valid")


def _model_labels(repo: str) -> set[str] | None:
    from huggingface_hub import hf_hub_download

    try:
        cfg = json.loads(Path(hf_hub_download(repo, "config.json")).read_text())
    except Exception:
        return None
    return {normalize(v) for v in (cfg.get("id2label") or {}).values()}


def _search_models(q: str, tag: str) -> list[Any]:
    from huggingface_hub import HfApi

    try:
        return list(HfApi().list_models(search=q, pipeline_tag=tag, sort="downloads", limit=15))
    except Exception:
        return []


def baseline_models(spec: TaskSpec, queries: list[str], k: int = 2, min_overlap: float = 0.6) -> list[tuple[str, str]]:
    tag = TASK_TAG.get((spec.modality, spec.task_type))
    if not tag or spec.task_type != "classify":
        return []
    ours = {normalize(n) for n in spec.label_names}
    seen, found = set(), []
    for q in queries:
        for m in _search_models(q, tag):
            if m.id in seen:
                continue
            seen.add(m.id)
            theirs = _model_labels(m.id)
            if theirs is None:
                continue
            cover = len(ours & theirs) / max(1, len(ours))
            if cover >= min_overlap:
                dl = getattr(m, "downloads", 0) or 0
                found.append((dl, m.id, f"{tag} model, {dl} downloads, covers {cover:.0%} of our labels"))
    found.sort(key=lambda t: -t[0])
    return [(repo, why) for _, repo, why in found[:k]]


PICK_BENCH = """You pick a public benchmark for a classifier. From the Hugging Face search results, choose the dataset \
that best matches the task and labels and has a held-out split (test or validation). Answer with its exact id, or "" if \
none fits. Prefer widely used, established benchmarks."""


def _pick_benchmark(spec: TaskSpec, hits: list[dict], cfg: dict | None) -> dict[str, Any]:
    listing = "\n".join(f"- {h['id']} (downloads={h['downloads']}, {', '.join(h['tags'])})" for h in hits)
    task = f"Task: {spec.summary or spec.description}\nModality: {spec.modality}\nLabels: {spec.label_names}"
    return get_llm("spec", cfg).json(
        PICK_BENCH,
        f"{task}\n\nResults:\n{listing}",
        {"type": "object", "properties": {"dataset": {"type": "string"}, "why": {"type": "string"}}},
    )


def _split_sources(
    spec: TaskSpec, src: HFSource, held: str, splits: list[str]
) -> tuple[HFSource, HFSource | None, str | None]:
    n_lab = len(spec.labels)
    ev = src.model_copy(update={"split": held, "sampling": "balanced", "per_label": max(50, 4000 // n_lab)})
    train_split = next((s for s in ("train",) if s in splits and s != held), None)
    if not train_split:
        return ev, None, None
    per_label = max(100, spec.data.train_size // n_lab)
    tr = src.model_copy(update={"split": train_split, "sampling": "balanced", "per_label": per_label})
    return ev, tr, train_split


def benchmark(
    spec: TaskSpec, queries: list[str], cfg: dict | None, log: Callable[[str], None]
) -> tuple[HFSource | None, HFSource | None, str]:
    from datasets import get_dataset_split_names

    hits = hf_source.search(queries)
    if not hits:
        return None, None, "no Hub results for the benchmark queries"
    out = _pick_benchmark(spec, hits, cfg)
    ds = out.get("dataset", "")
    if ds not in {h["id"] for h in hits}:
        return None, None, "no suitable benchmark among the search results"
    try:
        splits = get_dataset_split_names(ds)
    except Exception as e:
        return None, None, f"{ds}: cannot read splits ({type(e).__name__})"
    held = next((s for s in HELD_OUT_SPLITS if s in splits), None)
    if not held:
        return None, None, f"{ds} has no held-out split ({splits})"
    tmp = spec.model_copy(deep=True)
    tmp.data.hf_sources = [HFSource(dataset=ds, role="labeled", split=held)]
    mapped = hf_source.resolve(tmp, cfg, log)
    if not mapped:
        return None, None, f"{ds}: could not map its columns/labels onto ours"
    ev, tr, train_split = _split_sources(spec, mapped[0], held, splits)
    return ev, tr, f"{ds}: {out.get('why', '')} (eval on `{held}`" + (f", train on `{train_split}`)" if tr else ")")


def _add_benchmark(
    spec: TaskSpec, queries: list[str], cfg: dict | None, log: Callable[[str], None], decisions: list[dict]
) -> None:
    try:
        ev, tr, why = benchmark(spec, queries, cfg, log)
    except Exception as e:
        ev, tr, why = None, None, f"benchmark search failed: {type(e).__name__}: {e}"
    if ev:
        for e in spec.evals:
            e.primary = False
        spec.evals.append(EvalSuite(name="bench", kind="hf", source=ev, max_items=5000, primary=True))
        if tr:
            spec.data.hf_sources.append(tr)
    decisions.append({"step": "benchmark", "choice": ev.dataset if ev else None, "why": why})


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


def discover(
    spec: TaskSpec, queries: dict[str, list[str]], cfg: dict | None = None, log: Callable[[str], None] = print
) -> tuple[TaskSpec, list[dict]]:
    spec = spec.model_copy(deep=True)
    decisions: list[dict] = []
    if spec.task_type == "classify" and queries.get("benchmark"):
        _add_benchmark(spec, queries["benchmark"], cfg, log, decisions)
    bl = list(spec.baselines)
    _default_baselines(spec, bl, decisions)
    _model_baselines(spec, queries.get("baseline") or [], bl, decisions)
    spec.baselines = bl
    return spec, decisions
