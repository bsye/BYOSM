from __future__ import annotations

from collections.abc import Callable

from ..llm import get_llm, parallel_calls, pmap
from ..spec import TaskSpec
from .prompts import JUDGE_SYSTEM, task_context

JUDGE_ROLE = "judge"
JUDGE_BATCH = 40
DROP = "__drop__"


def _schema(spec: TaskSpec) -> dict:
    verdict = {"n": {"type": "integer"}, "label": {"type": "string", "enum": [*spec.label_names, DROP]}}
    return {
        "type": "object",
        "properties": {"labels": {"type": "array", "items": {"type": "object", "properties": verdict}}},
    }


def judge_labels(
    spec: TaskSpec,
    texts: list[str],
    cfg: dict | None = None,
    workers: int | None = None,
    log: Callable[[str], None] = print,
) -> list[str | None]:
    if not texts:
        return []
    llm, schema, answers = get_llm(JUDGE_ROLE, cfg), _schema(spec), {*spec.label_names, DROP}

    def run(start: int) -> tuple[int, list[str | None]]:
        chunk = texts[start : start + JUDGE_BATCH]
        listing = "\n".join(f"{i}. {t}" for i, t in enumerate(chunk, 1))
        ask = f"Return {len(chunk)} labels, each with its input's number n."
        out = llm.json(JUDGE_SYSTEM, f"{task_context(spec)}\n\nInputs:\n{listing}\n\n{ask}", schema)
        labs = {v.get("n"): v.get("label") for v in out.get("labels", []) if isinstance(v, dict)}
        return start, [labs.get(i) if labs.get(i) in answers else None for i in range(1, len(chunk) + 1)]

    found: list[str | None] = [None] * len(texts)
    starts = range(0, len(texts), JUDGE_BATCH)
    for start, labs in pmap(
        run, starts, workers or parallel_calls(cfg), on_error=lambda e: log(f"  judge batch failed: {e}")
    ):
        found[start : start + len(labs)] = labs
    return found


def judge(
    spec: TaskSpec,
    rows: list[dict],
    cfg: dict | None = None,
    workers: int | None = None,
    log: Callable[[str], None] = print,
) -> list[dict]:
    if spec.task_type == "frame" or not rows:
        return rows
    labels = judge_labels(spec, [r["input"] for r in rows], cfg, workers, log)
    kept = [r for r, lab in zip(rows, labels, strict=True) if lab == r["label"]]
    log(f"  judge kept {len(kept)}/{len(rows)} ({len(kept) / max(1, len(rows)):.0%} agreement)")
    return kept
