"""LLM-written text data: diverse generation, blind second-opinion labeling, adversarial held-out sets."""
from __future__ import annotations

import random
from typing import Any, Callable

from ..llm import get_llm, pmap
from ..spec import TaskSpec
from .store import dedupe, record

GEN_SYSTEM = """You write training data for a tiny, single-purpose classifier. Inputs must look exactly like what the \
deployed model will receive in production: realistic, varied, messy where reality is messy (typos, fragments, slang, \
code-switching, missing punctuation) and never templated. Vary length, structure, vocabulary and openings aggressively. \
Every label you assign must be correct under the definitions; when in doubt, choose the label the definitions dictate, \
not intuition."""

EVAL_SYSTEM = """You are an independent evaluator writing a HELD-OUT test set to find where a tiny classifier fails. \
You did not write its training data. Write inputs as real users/systems would produce them, with a realistic share of \
hard cases: confusers, near-misses, ambiguity resolved by the definitions, rare phrasings, long-tail domains, adversarial \
but plausible wording. Never write obviously synthetic or templated text. Every label must be correct under the definitions."""

JUDGE_SYSTEM = """You are a meticulous annotator. Label each input strictly according to the label definitions. \
Return exactly one label per input, in order. If an input is unusable (not a plausible input, or genuinely ambiguous \
even under the definitions) label it "__drop__"."""


def _label_block(spec: TaskSpec) -> str:
    return "\n".join(f'- "{l.name}": {l.definition}' for l in spec.labels)


def _context(spec: TaskSpec) -> str:
    return (
        f"Task: {spec.summary or spec.description}\nInputs: {spec.input_description}\nLanguages: {', '.join(spec.languages)}\n"
        f"Labels:\n{_label_block(spec)}"
    )


def _schema(spec: TaskSpec) -> dict:
    if spec.task_type == "frame":
        item = {"type": "object", "properties": {"words": {"type": "array", "items": {"type": "string"}}, "labels": {"type": "array", "items": {"type": "string", "enum": spec.label_names}}}}
    else:
        item = {"type": "object", "properties": {"text": {"type": "string"}, "label": {"type": "string", "enum": spec.label_names}}}
    return {"type": "object", "properties": {"examples": {"type": "array", "items": item}}}


def _to_records(spec: TaskSpec, out: dict, source: str, **meta: Any) -> list[dict]:
    rows = []
    names = set(spec.label_names)
    for ex in out.get("examples", []):
        if spec.task_type == "frame":
            w, l = ex.get("words") or [], ex.get("labels") or []
            if w and len(w) == len(l) and all(x in names for x in l) and all(" " not in t for t in w):
                rows.append(record(" ".join(w), l, source, **meta))
        else:
            t, l = (ex.get("text") or "").strip(), ex.get("label")
            if t and l in names:
                rows.append(record(t, l, source, **meta))
    return rows


def _jobs(spec: TaskSpec, n: int, rng: random.Random, adversarial: bool) -> list[dict]:
    k = spec.data.per_call
    jobs = []
    for i in range(max(1, -(-n // k))):
        label = spec.label_names[i % len(spec.label_names)]
        combo = {a.name: rng.choice(a.values) for a in spec.diversity_axes if a.values}
        hard = rng.sample(spec.hard_cases, min(len(spec.hard_cases), 4 if adversarial else 2))
        jobs.append({"label": label, "combo": combo, "hard": hard, "k": k, "seed": rng.randint(0, 10**6)})
    return jobs


def _gen_prompt(spec: TaskSpec, job: dict, adversarial: bool, avoid: list[str]) -> str:
    combo = "; ".join(f"{k}={v}" for k, v in job["combo"].items()) or "free"
    hard = "\n".join(f"- {h}" for h in job["hard"]) or "- (none)"
    avoid_s = "\n".join(f"- {a[:160]}" for a in avoid) or "- (none)"
    if spec.task_type == "frame":
        target = f'Write {job["k"]} inputs, each as a list of whitespace-free words with one label per word. Make sure label "{job["label"]}" appears in most of them.'
    elif adversarial:
        target = f'Write {job["k"]} examples. About half should truly be "{job["label"]}"; the rest should be the hardest realistic confusers for it (labeled with their true label).'
    else:
        target = f'Write {job["k"]} examples. About 70% labeled "{job["label"]}", ~30% near-misses that look like "{job["label"]}" but truly belong to another label (labeled correctly).'
    return (
        f"{_context(spec)}\n\nSituation for this batch: {combo}\n\nHard cases to weave in where natural:\n{hard}\n\n"
        f"Do NOT resemble these existing examples (style, opening, structure):\n{avoid_s}\n\n{target}\n(variation seed {job['seed']})"
    )


def judge(spec: TaskSpec, rows: list[dict], cfg: dict | None = None, concurrency: int = 8, log: Callable[[str], None] = print) -> list[dict]:
    """Blind second-opinion labels; keep only agreements. Cheap insurance against label noise, which caps precision."""
    if spec.task_type == "frame" or not rows:
        return rows
    llm = get_llm("judge", cfg)
    schema = {"type": "object", "properties": {"labels": {"type": "array", "items": {"type": "string", "enum": [*spec.label_names, "__drop__"]}}}}
    chunks = [rows[i : i + 40] for i in range(0, len(rows), 40)]

    def run(chunk: list[dict]) -> list[dict]:
        listing = "\n".join(f"{i + 1}. {r['input']}" for i, r in enumerate(chunk))
        out = llm.json(JUDGE_SYSTEM, f"{_context(spec)}\n\nInputs:\n{listing}\n\nReturn {len(chunk)} labels.", schema)
        labs = out.get("labels", [])
        if len(labs) != len(chunk):
            return []
        return [r for r, l in zip(chunk, labs) if l == r["label"]]

    kept = [r for part in pmap(run, chunks, concurrency, on_error=lambda e: log(f"  judge batch failed: {e}")) for r in part]
    log(f"  judge kept {len(kept)}/{len(rows)} ({len(kept) / max(1, len(rows)):.0%} agreement)")
    return kept


def generate(
    spec: TaskSpec,
    n: int,
    split: str,
    cfg: dict | None = None,
    against: list[dict] | None = None,
    jobs: list[dict] | None = None,
    log: Callable[[str], None] = print,
) -> list[dict]:
    concurrency = (cfg or {}).get("concurrency", 8)
    adversarial = split in ("dev", "test")
    llm = get_llm("eval" if adversarial else "generate", cfg)
    rng = random.Random(f"{spec.name}-{split}-{len(against or [])}")
    jobs = jobs or _jobs(spec, n, rng, adversarial)
    pool = [r["input"] for r in (against or [])]
    system = EVAL_SYSTEM if adversarial else GEN_SYSTEM
    schema = _schema(spec)
    log(f"  generating {split}: {len(jobs)} calls x {spec.data.per_call} with {llm.name}")

    def run(job: dict) -> list[dict]:
        avoid = random.Random(job["seed"]).sample(pool, min(8, len(pool)))
        prompt = job.get("prompt") or _gen_prompt(spec, job, adversarial, avoid)
        return _to_records(spec, llm.json(system, prompt, schema), f"llm:{llm.name}", split=split, combo=job.get("combo"))

    rows = [r for part in pmap(run, jobs, concurrency, on_error=lambda e: log(f"  generation call failed: {e}")) for r in part]
    rows = dedupe(rows, against=against)
    if spec.data.judge:
        rows = judge(spec, rows, cfg, concurrency, log)
    return rows


ANALYZE_SYSTEM = """You analyze the errors of a tiny classifier on a dev set and design targeted training data to fix them. \
Find the systematic failure patterns (not one-off noise). For each, explain the confusion and say precisely what new \
training examples would teach the boundary. Also flag dev examples whose gold label looks wrong under the definitions."""


def refine_jobs(spec: TaskSpec, errors: list[dict], n: int, cfg: dict | None = None) -> tuple[list[dict], dict]:
    """Ask the analyst LLM for failure patterns on dev errors, then turn them into generation jobs."""
    schema = {
        "type": "object",
        "properties": {
            "patterns": {"type": "array", "items": {"type": "object", "properties": {
                "name": {"type": "string"}, "explanation": {"type": "string"},
                "data_recipe": {"type": "string"}, "share": {"type": "number"}}}},
            "suspect_gold_ids": {"type": "array", "items": {"type": "string"}},
        },
    }
    listing = "\n".join(f'[{e["id"]}] gold={e["gold"]} pred={e["pred"]} p={e["confidence"]:.2f}: {str(e["input"])[:300]}' for e in errors[:150])
    out = get_llm("analyze", cfg).json(ANALYZE_SYSTEM, f"{_context(spec)}\n\nDev errors ({len(errors)} total):\n{listing}", schema)
    pats = [p for p in out.get("patterns", []) if p.get("data_recipe")]
    total = sum(max(0.05, p.get("share", 1)) for p in pats) or 1
    jobs, k = [], spec.data.per_call
    rng = random.Random(len(errors))
    for p in pats:
        calls = max(1, round(n * max(0.05, p.get("share", 1)) / total / k))
        for _ in range(calls):
            combo = {a.name: rng.choice(a.values) for a in spec.diversity_axes if a.values}
            jobs.append({"combo": combo, "seed": rng.randint(0, 10**6), "prompt": (
                f"{_context(spec)}\n\nThe model systematically fails on this pattern:\n{p['name']}: {p['explanation']}\n\n"
                f"Write {k} new, varied training examples following this recipe: {p['data_recipe']}\n"
                f"Situation: {'; '.join(f'{a}={v}' for a, v in combo.items())}. Include both sides of the boundary, labeled correctly.")})
    return jobs, out
