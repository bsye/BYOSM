"""Discovery: given a compiled spec, find on the Hugging Face Hub
  - a public benchmark to be the primary eval suite (its held-out split), and its train split as training data
  - existing models that solve the same task, as baselines (labels matched to ours by name)
and add the zero-shot LLM baseline(s). Every choice is returned as a logged decision with its reason.
"""
from __future__ import annotations

from .text import normalize

import json
from typing import Callable

from .data import hf_source
from .llm import get_llm
from .spec import EvalSuite, HFSource, TaskSpec

TASK_TAG = {("text", "classify"): "text-classification", ("image", "classify"): "image-classification",
            ("document", "classify"): "image-classification", ("audio", "classify"): "audio-classification",
            ("audio", "frame"): "voice-activity-detection", ("text", "frame"): "token-classification"}



def baseline_models(spec: TaskSpec, queries: list[str], k: int = 2, min_overlap: float = 0.6) -> list[tuple[str, str]]:
    """HF models for this task type whose labels cover >= min_overlap of ours. -> [(repo, reason)]"""
    from huggingface_hub import HfApi, hf_hub_download

    tag = TASK_TAG.get((spec.modality, spec.task_type))
    if not tag or spec.task_type != "classify":
        return []
    ours = {normalize(n) for n in spec.label_names}
    seen, found = set(), []
    for q in queries:
        try:
            models = list(HfApi().list_models(search=q, pipeline_tag=tag, sort="downloads", limit=15))
        except Exception:
            continue
        for m in models:
            if m.id in seen:
                continue
            seen.add(m.id)
            try:
                cfg = json.loads(open(hf_hub_download(m.id, "config.json")).read())
            except Exception:
                continue
            theirs = {normalize(v) for v in (cfg.get("id2label") or {}).values()}
            cover = len(ours & theirs) / max(1, len(ours))
            if cover >= min_overlap:
                dl = getattr(m, "downloads", 0) or 0
                found.append((dl, m.id, f"{tag} model, {dl} downloads, covers {cover:.0%} of our labels"))
    found.sort(key=lambda t: -t[0])
    return [(repo, why) for _, repo, why in found[:k]]


PICK_BENCH = """You pick a public benchmark for a classifier. From the Hugging Face search results, choose the dataset \
that best matches the task and labels and has a held-out split (test or validation). Answer with its exact id, or "" if \
none fits. Prefer widely used, established benchmarks."""


def benchmark(spec: TaskSpec, queries: list[str], cfg: dict | None, log: Callable[[str], None]) -> tuple[HFSource | None, HFSource | None, str]:
    """-> (eval source on the held-out split, train source on the train split, reason)"""
    from datasets import get_dataset_split_names

    hits = hf_source.search(queries)
    if not hits:
        return None, None, "no Hub results for the benchmark queries"
    listing = "\n".join(f"- {h['id']} (downloads={h['downloads']}, {', '.join(h['tags'])})" for h in hits)
    out = get_llm("spec", cfg).json(PICK_BENCH, f"Task: {spec.summary or spec.description}\nModality: {spec.modality}\nLabels: {spec.label_names}\n\nResults:\n{listing}",
                                    {"type": "object", "properties": {"dataset": {"type": "string"}, "why": {"type": "string"}}})
    ds = out.get("dataset", "")
    if ds not in {h["id"] for h in hits}:
        return None, None, "no suitable benchmark among the search results"
    try:
        splits = get_dataset_split_names(ds)
    except Exception as e:  # noqa: BLE001
        return None, None, f"{ds}: cannot read splits ({type(e).__name__})"
    held = next((s for s in ("test", "validation", "val", "valid") if s in splits), None)
    if not held:
        return None, None, f"{ds} has no held-out split ({splits})"
    tmp = spec.model_copy(deep=True)
    tmp.data.hf_sources = [HFSource(dataset=ds, role="labeled", split=held)]
    mapped = hf_source.resolve(tmp, cfg, log)
    if not mapped:
        return None, None, f"{ds}: could not map its columns/labels onto ours"
    src = mapped[0]
    n_lab = len(spec.labels)
    # ~4000 items -> ~800 calib: selection compares models on calib, and ~200 items can't separate a 2-pt difference
    ev = src.model_copy(update={"split": held, "sampling": "balanced", "per_label": max(50, 4000 // n_lab)})
    train_split = next((s for s in ("train",) if s in splits and s != held), None)
    tr = src.model_copy(update={"split": train_split, "sampling": "balanced", "per_label": max(100, spec.data.train_size // n_lab)}) if train_split else None
    return ev, tr, f"{ds}: {out.get('why', '')} (eval on `{held}`" + (f", train on `{train_split}`)" if tr else ")")


def discover(spec: TaskSpec, queries: dict[str, list[str]], cfg: dict | None = None, log: Callable[[str], None] = print) -> tuple[TaskSpec, list[dict]]:
    spec = spec.model_copy(deep=True)
    decisions: list[dict] = []
    # Benchmark suite (+ its train split as real training data).
    if spec.task_type == "classify" and queries.get("benchmark"):
        try:
            ev, tr, why = benchmark(spec, queries["benchmark"], cfg, log)
        except Exception as e:  # noqa: BLE001
            ev, tr, why = None, None, f"benchmark search failed: {type(e).__name__}: {e}"
        if ev:
            for e in spec.evals:
                e.primary = False
            spec.evals.append(EvalSuite(name="bench", kind="hf", source=ev, max_items=5000, primary=True))
            if tr:
                spec.data.hf_sources.append(tr)
        decisions.append({"step": "benchmark", "choice": ev.dataset if ev else None, "why": why})
    # Baselines: the LLM(s) + existing task models.
    bl = list(spec.baselines)
    if spec.modality in ("text", "document", "image") and spec.task_type == "classify" and "llm" not in bl:
        bl.append("llm")
        decisions.append({"step": "baseline", "choice": "llm", "why": "the zero-shot LLM is the System-2 bar the specialist must beat"})
    if spec.modality == "audio" and spec.task_type == "frame":
        for b in ("energy", "silero"):
            if b not in bl:
                bl.append(b)
        decisions.append({"step": "baseline", "choice": "energy, silero", "why": "frame-level audio: a sanity floor + LiveKit's VAD"})
    try:
        for repo, why in baseline_models(spec, queries.get("baseline") or []):
            if f"hf:{repo}" not in bl:
                bl.append(f"hf:{repo}")
                decisions.append({"step": "baseline", "choice": f"hf:{repo}", "why": why})
    except Exception as e:  # noqa: BLE001
        decisions.append({"step": "baseline", "choice": None, "why": f"model search failed: {type(e).__name__}"})
    spec.baselines = bl
    return spec, decisions
