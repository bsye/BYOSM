from __future__ import annotations

import itertools
import random
from collections.abc import Callable
from typing import Any

from ..events import emit
from ..llm import get_llm, pmap
from ..spec import TaskSpec
from .prompts import ANALYZE_SYSTEM, EVAL_SYSTEM, GEN_SYSTEM, JUDGE_SYSTEM, REFINE_SCHEMA
from .store import dedupe, record

DEFAULT_CONCURRENCY = 8


def concurrency(cfg: dict | None) -> int:
    return (cfg or {}).get("concurrency", DEFAULT_CONCURRENCY)


def role(split: str) -> str:
    return "eval" if split in ("dev", "test") else "generate"


def calls(n: int, per_call: int) -> int:
    return max(1, -(-n // per_call))


def _label_block(spec: TaskSpec, names: list[str] | None = None) -> str:
    return "\n".join(f'- "{lab.name}": {lab.definition}' for lab in spec.labels if names is None or lab.name in names)


def _context(spec: TaskSpec, names: list[str] | None = None) -> str:
    return (
        f"Task: {spec.summary or spec.description}\nInputs: {spec.input_description}\nLanguages: {', '.join(spec.languages)}\n"
        f"Labels:\n{_label_block(spec, names)}"
    )


def _schema(spec: TaskSpec) -> dict:
    if spec.task_type == "frame":
        item = {
            "type": "object",
            "properties": {
                "words": {"type": "array", "items": {"type": "string"}},
                "labels": {"type": "array", "items": {"type": "string", "enum": spec.label_names}},
            },
        }
    else:
        item = {
            "type": "object",
            "properties": {"text": {"type": "string"}, "label": {"type": "string", "enum": spec.label_names}},
        }
    return {"type": "object", "properties": {"examples": {"type": "array", "items": item}}}


def _to_records(spec: TaskSpec, out: dict, source: str, **meta: Any) -> list[dict]:
    rows = []
    names = set(spec.label_names)
    for ex in out.get("examples", []):
        if spec.task_type == "frame":
            w, labs = ex.get("words") or [], ex.get("labels") or []
            if w and len(w) == len(labs) and all(x in names for x in labs) and all(" " not in t for t in w):
                rows.append(record(" ".join(w), labs, source, **meta))
        else:
            t, lab = (ex.get("text") or "").strip(), ex.get("label")
            if t and lab in names:
                rows.append(record(t, lab, source, **meta))
    return rows


def _jobs(spec: TaskSpec, n: int, rng: random.Random, adversarial: bool, labels: list[str] | None = None) -> list[dict]:
    k = spec.data.per_call
    labels = labels or spec.label_names
    jobs = []
    for i in range(calls(n, k)):
        label = labels[i % len(labels)]
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


def judge(
    spec: TaskSpec, rows: list[dict], cfg: dict | None = None, concurrency: int = 8, log: Callable[[str], None] = print
) -> list[dict]:
    if spec.task_type == "frame" or not rows:
        return rows
    llm = get_llm("judge", cfg)
    verdict = {"n": {"type": "integer"}, "label": {"type": "string", "enum": [*spec.label_names, "__drop__"]}}
    schema = {
        "type": "object",
        "properties": {"labels": {"type": "array", "items": {"type": "object", "properties": verdict}}},
    }
    chunks = [rows[i : i + 40] for i in range(0, len(rows), 40)]

    def run(chunk: list[dict]) -> list[dict]:
        listing = "\n".join(f"{i + 1}. {r['input']}" for i, r in enumerate(chunk))
        ask = f"Return {len(chunk)} labels, each with its input's number n."
        out = llm.json(JUDGE_SYSTEM, f"{_context(spec)}\n\nInputs:\n{listing}\n\n{ask}", schema)
        labs = {v.get("n"): v.get("label") for v in out.get("labels", []) if isinstance(v, dict)}
        return [r for i, r in enumerate(chunk, 1) if labs.get(i) == r["label"]]

    kept = [
        r for part in pmap(run, chunks, concurrency, on_error=lambda e: log(f"  judge batch failed: {e}")) for r in part
    ]
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
    labels: list[str] | None = None,
    on_rows: Callable[[list[dict]], None] | None = None,
) -> list[dict]:
    workers = concurrency(cfg)
    adversarial = role(split) == "eval"
    llm = get_llm(role(split), cfg)
    rng = random.Random(f"{spec.name}-{split}-{len(against or [])}")
    jobs = jobs or _jobs(spec, n, rng, adversarial, labels)
    pool = [r["input"] for r in (against or [])]
    system = EVAL_SYSTEM if adversarial else GEN_SYSTEM
    log(f"  generating {split}: {len(jobs)} calls x {spec.data.per_call} with {llm.name}")
    done = itertools.count(1)

    def run(job: dict) -> list[dict]:
        avoid = random.Random(job["seed"]).sample(pool, min(8, len(pool)))
        prompt = job.get("prompt") or _gen_prompt(spec, job, adversarial, avoid)
        out = _to_records(
            spec, llm.json(system, prompt, _schema(spec)), f"llm:{llm.name}", split=split, combo=job.get("combo")
        )
        if on_rows:
            on_rows(out)
        emit("writing", split=split, done=next(done), total=len(jobs), specialist=spec.name)
        return out

    rows = [
        r for part in pmap(run, jobs, workers, on_error=lambda e: log(f"  generation call failed: {e}")) for r in part
    ]
    rows = dedupe(rows, against=against)
    if spec.data.judge:
        rows = judge(spec, rows, cfg, workers, log)
    return rows


def _pattern_prompt(spec: TaskSpec, p: dict, k: int, combo: dict) -> str:
    situation = "; ".join(f"{a}={v}" for a, v in combo.items())
    return (
        f"{_context(spec)}\n\nThe model systematically fails on this pattern:\n{p['name']}: {p['explanation']}\n\n"
        f"Write {k} new, varied training examples following this recipe: {p['data_recipe']}\n"
        f"Situation: {situation}. Include both sides of the boundary, labeled correctly."
    )


def _pattern_jobs(spec: TaskSpec, pats: list[dict], n: int, rng: random.Random) -> list[dict]:
    total = sum(max(0.05, p.get("share", 1)) for p in pats) or 1
    jobs, k = [], spec.data.per_call
    for p in pats:
        calls = max(1, round(n * max(0.05, p.get("share", 1)) / total / k))
        for _ in range(calls):
            combo = {a.name: rng.choice(a.values) for a in spec.diversity_axes if a.values}
            seed = rng.randint(0, 10**6)
            jobs.append({"combo": combo, "seed": seed, "prompt": _pattern_prompt(spec, p, k, combo)})
    return jobs


def refine_jobs(spec: TaskSpec, errors: list[dict], n: int, cfg: dict | None = None) -> tuple[list[dict], dict]:
    listing = "\n".join(
        f"[{e['id']}] gold={e['gold']} pred={e['pred']} p={e['confidence']:.2f}: {str(e['input'])[:300]}"
        for e in errors[:150]
    )
    prompt = f"{_context(spec)}\n\nDev errors ({len(errors)} total):\n{listing}"
    out = get_llm("analyze", cfg).json(ANALYZE_SYSTEM, prompt, REFINE_SCHEMA)
    pats = [p for p in out.get("patterns", []) if p.get("data_recipe")]
    return _pattern_jobs(spec, pats, n, random.Random(len(errors))), out
