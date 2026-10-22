from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from ..project import Project
from ..spec import TaskSpec
from .table import LABEL_COL, columns, input_column, write_template

REQUEST_DIR = "data_request"
DATA_REQUEST = "data-request"
Z95 = 1.96
CI_HALF_WIDTH = 0.03
SIZING_ACCURACY = 0.9
MIN_TEST_PER_LABEL = 20
TRAIN_FLOOR = 50
TRAIN_RECOMMENDED = 200
CONFUSABLE_FACTOR = 2
EXAMPLES_PER_LABEL = 2

PLAN_SYSTEM = """You help a user collect labelled data for a small classifier. Given the task and its labels, list the \
label pairs a model is most likely to confuse and why, one short tip per label on its borderline cases, and \
{n} realistic example inputs per label (for documents: a plausible PDF file name; for audio: a .wav file name)."""
TAGGING_NOTE = " Labels are word tags: write each example as a sentence with its spans marked [type : words]."


def missing_labels(spec: TaskSpec) -> list[str]:
    if (spec.modality == "text" and spec.data.synthetic) or _has_local(spec):
        return []
    labeled = [s for s in spec.data.hf_sources if s.role == "labeled"]
    if any(not s.label_map for s in labeled):
        return []
    covered = {b for s in labeled for _, b in s.label_map}
    return [n for n in spec.label_names if n not in covered]


def _has_local(spec: TaskSpec) -> bool:
    return bool(spec.data.local_dir) and Path(spec.data.local_dir).exists()


def score_rows_needed(target: float, n_labels: int) -> tuple[int, int]:
    p = min(target, SIZING_ACCURACY)
    total = math.ceil(Z95**2 * p * (1 - p) / CI_HALF_WIDTH**2)
    return total, max(MIN_TEST_PER_LABEL, math.ceil(total / n_labels))


def _array(item: dict) -> dict:
    return {"type": "array", "items": item}


def _plan_schema(names: list[str]) -> dict:
    label = {"type": "string", "enum": names}
    pair = {"type": "object", "properties": {"a": label, "b": label, "why": {"type": "string"}}}
    tip = {"type": "object", "properties": {"label": label, "tip": {"type": "string"}}}
    example = {"type": "object", "properties": {"label": label, "input": {"type": "string"}}}
    props = {"confusable": _array(pair), "tips": _array(tip), "examples": _array(example)}
    return {"type": "object", "properties": props}


def _advice(spec: TaskSpec, cfg: dict | None) -> dict[str, Any]:
    from ..llm import get_llm

    labels = "\n".join(f"- {lab.name}: {lab.definition}" for lab in spec.labels)
    prompt = f"Task: {spec.description}\nInput: {spec.modality}\nLabels:\n{labels}"
    system = PLAN_SYSTEM.format(n=EXAMPLES_PER_LABEL) + (TAGGING_NOTE if spec.task_type == "frame" else "")
    return get_llm("spec", cfg).json(system, prompt, _plan_schema(spec.label_names))


def counts(spec: TaskSpec, confusable: list[dict]) -> dict[str, dict[str, int]]:
    hard = {p[k] for p in confusable for k in ("a", "b")}
    _, test = score_rows_needed(spec.objective.target, len(spec.labels))
    out = {}
    for n in spec.label_names:
        f = CONFUSABLE_FACTOR if n in hard else 1
        out[n] = {"train_min": TRAIN_FLOOR * f, "train_recommended": TRAIN_RECOMMENDED * f, "test_needed": test}
    return out


def _example_rows(spec: TaskSpec, examples: list[dict]) -> list[list[str]]:
    cols = columns(spec)
    rows = []
    for e in examples:
        cell = {input_column(spec): e.get("input", ""), LABEL_COL: e.get("label", "")}
        rows.append([cell.get(c, "") for c in cols])
    return rows


def _guide(spec: TaskSpec, need: dict, tips: dict[str, str]) -> list[list[Any]]:
    head = ["label", "definition", "borderline cases", "train rows (min)", "train rows (recommended)", "test rows"]
    rows = [[n.name, n.definition, tips.get(n.name, "")] + list(need[n.name].values()) for n in spec.labels]
    return [head, *rows]


def _markdown(spec: TaskSpec, reason: str, need: dict, advice: dict, files: list[Path]) -> str:
    total, _ = score_rows_needed(spec.objective.target, len(spec.labels))
    lines = [
        f"# Data needed for {spec.name}",
        "",
        f"{reason}",
        "",
        f"Fill in `{files[0].name}` or `{files[1].name}` (one row per example) and run "
        f"`byosm check-data {spec.name} <file>`.",
        "",
        "| label | train rows (min) | train rows (recommended) | test rows |",
        "|---|---|---|---|",
        *(f"| {n} | {c['train_min']} | {c['train_recommended']} | {c['test_needed']} |" for n, c in need.items()),
        "",
        f"Why these numbers: measuring accuracy to within {CI_HALF_WIDTH * 100:.0f} points (95% CI) needs about "
        f"{total} test rows overall, at least {MIN_TEST_PER_LABEL} per label. Training starts at {TRAIN_FLOOR} rows per "
        f"label with a pretrained backbone; labels that are easy to confuse need {CONFUSABLE_FACTOR}x. After the first "
        "build, `byosm improve` reads the learning curve and says whether more rows would help.",
        "",
        "Mark a row `test` in the split column to keep it out of training; blank rows are split for you.",
    ]
    pairs = advice.get("confusable", [])
    if pairs:
        lines += ["", "Easy to confuse:", *(f"- {p['a']} vs {p['b']}: {p['why']}" for p in pairs)]
    return "\n".join(lines) + "\n"


def write_request(p: Project, spec: TaskSpec, reason: str, cfg: dict | None = None) -> Path:
    advice = _advice(spec, cfg)
    need = counts(spec, advice.get("confusable", []))
    tips = {t["label"]: t["tip"] for t in advice.get("tips", [])}
    out = p.root / REQUEST_DIR
    files = write_template(
        out, columns(spec), _example_rows(spec, advice.get("examples", [])), _guide(spec, need, tips)
    )
    markdown = _markdown(spec, reason, need, advice, files)
    p.save_report(DATA_REQUEST, {"reason": reason, "counts": need, "advice": advice, "markdown": markdown})
    return out
