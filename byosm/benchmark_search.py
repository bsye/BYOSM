from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from .data import hf_source
from .llm import get_llm, role_model
from .spec import EvalSuite, HFSource, TaskSpec
from .text import slug

HELD_OUT_SPLITS = ("test", "validation", "val", "valid")
PICKER_ROLE = "spec"
RULE = "rule"
OTHER_HITS = 3
BENCH_SUITE = "bench"
BENCH_MAX_ITEMS = 5000
EVAL_ROWS = 4000
MIN_EVAL_PER_LABEL = 50
MIN_TRAIN_PER_LABEL = 100

PICK_BENCH = """You pick a public benchmark for a classifier. From the Hugging Face search results, choose the dataset \
that best matches the task and labels. Prefer widely used, established, labelled datasets. Answer with its exact id, \
or "" if none fits, and say why in one sentence."""


def try_sources(name: str, sources: list[dict], tag: str) -> list[str]:
    return ["exp", "create", name, f"try-{slug(tag)}", "--set", f"data.hf_sources={json.dumps(sources)}", "--run"]


def try_setting(name: str, path: str, value: Any, tag: str) -> list[str]:
    return ["exp", "create", name, f"try-{slug(tag)}", "--set", f"{path}={json.dumps(value)}", "--run"]


def _dataset_alternatives(spec: TaskSpec, hits: list[dict], chosen: str, by: str) -> list[dict]:
    others = sorted((h for h in hits if h["id"] != chosen), key=lambda h: -h["downloads"])[:OTHER_HITS]
    return [
        {
            "option": h["id"],
            "why_not": f"{by} preferred {chosen or 'none of them'} ({h['downloads']} downloads)",
            "args": try_sources(spec.name, [{"dataset": h["id"], "role": "labeled"}], h["id"].split("/")[-1]),
        }
        for h in others
    ]


def _pick(spec: TaskSpec, hits: list[dict], cfg: dict | None) -> dict[str, Any]:
    listing = "\n".join(f"- {h['id']} (downloads={h['downloads']}, {', '.join(h['tags'])})" for h in hits)
    task = f"Task: {spec.summary or spec.description}\nModality: {spec.modality}\nLabels: {spec.label_names}"
    schema = {"type": "object", "properties": {"dataset": {"type": "string"}, "why": {"type": "string"}}}
    return get_llm(PICKER_ROLE, cfg).json(PICK_BENCH, f"{task}\n\nResults:\n{listing}", schema)


def _balanced(src: HFSource, per_label: int, split: str | None = None) -> HFSource:
    return src.model_copy(
        update={"sampling": "balanced", "per_label": per_label, **({"split": split} if split else {})}
    )


def _sources(spec: TaskSpec, ds: str, splits: list[str], cfg: dict | None, log: Callable) -> dict[str, Any]:
    n = len(spec.labels)
    held = next((s for s in HELD_OUT_SPLITS if s in splits), None)
    tmp = spec.model_copy(deep=True)
    tmp.data.hf_sources = [HFSource(dataset=ds, role="labeled", split=held or s) for s in ([held] if held else splits)]
    mapped = hf_source.resolve(tmp, cfg, log)
    if not mapped:
        return {"reason": f"{ds}: could not map its columns or labels onto ours"}
    train_per_label = max(MIN_TRAIN_PER_LABEL, spec.data.train_size // n)
    if not held:
        why = f"no test split ({', '.join(splits)}), so it trains on every split and part is held out as the test set"
        return {"train": [_balanced(m, train_per_label) for m in mapped], "reason": why}
    ev = _balanced(mapped[0], max(MIN_EVAL_PER_LABEL, EVAL_ROWS // n), held)
    train = [_balanced(mapped[0], train_per_label, "train")] if "train" in splits and held != "train" else []
    return {"eval": ev, "train": train, "reason": f"evaluate on `{held}`" + (", train on `train`" if train else "")}


def benchmark(spec: TaskSpec, queries: list[str], cfg: dict | None, log: Callable) -> dict[str, Any]:
    from datasets import get_dataset_split_names

    hits = hf_source.search(queries)
    if not hits:
        return {"reason": "no Hub results for the benchmark queries", "by": RULE, "alternatives": []}
    out, by = _pick(spec, hits, cfg), role_model(PICKER_ROLE, cfg)
    ds = out.get("dataset", "")
    alts = _dataset_alternatives(spec, hits, ds, by)
    if ds not in {h["id"] for h in hits}:
        return {"reason": "no suitable benchmark among the search results", "by": by, "alternatives": alts}
    try:
        splits = get_dataset_split_names(ds)
    except Exception as e:
        return {
            "dataset": ds,
            "reason": f"{ds}: cannot read splits ({type(e).__name__})",
            "by": RULE,
            "alternatives": alts,
        }
    found = _sources(spec, ds, splits, cfg, log)
    why = f"{ds}: {out.get('why', '')} ({found['reason']})"
    return {"dataset": ds, **found, "reason": why, "by": by, "alternatives": alts}


def add_benchmark(spec: TaskSpec, queries: list[str], cfg: dict | None, log: Callable, decisions: list[dict]) -> None:
    try:
        found = benchmark(spec, queries, cfg, log)
    except Exception as e:
        found = {"reason": f"benchmark search failed: {type(e).__name__}: {e}", "by": RULE, "alternatives": []}
    ev, train = found.get("eval"), found.get("train") or []
    if ev:
        for e in spec.evals:
            e.primary = False
        spec.evals.append(EvalSuite(name=BENCH_SUITE, kind="hf", source=ev, max_items=BENCH_MAX_ITEMS, primary=True))
    spec.data.hf_sources.extend(train)
    used = ev or train
    decisions.append(
        {
            "step": "benchmark",
            "choice": found.get("dataset") if used else None,
            "why": found["reason"],
            "by": found["by"],
            "alternatives": found["alternatives"],
        }
    )
    if train:
        prefer_real_rows(spec, decisions)


def prefer_real_rows(spec: TaskSpec, decisions: list[dict]) -> None:
    if not spec.data.synthetic:
        return
    spec.data.synthetic = False
    alt = {
        "option": "also write training data with the LLM",
        "why_not": "more rows, but the model also learns the LLM's style and mistakes",
        "args": try_setting(spec.name, "data.synthetic", True, "llm-data"),
    }
    why = "real labelled rows were found, so no LLM-written rows are needed"
    decisions.append({"step": "data", "choice": "real rows only", "why": why, "by": RULE, "alternatives": [alt]})
