from __future__ import annotations

import json
from typing import Any

from ..metrics import metric_label
from ..model_refs import model_ref
from . import sections
from .facts import CardFacts, suite_dataset

CARD_ROLE = "card"
SAME_AS_SECTION = {"usage", "reproduce"}
PLACES = {"RESULTS": "results", "TEST_SETS": "test_sets", "USAGE": "usage", "REPRODUCE": "reproduce"}
FREE_INTEGERS = 10
TOLERANCE = 0.006
SCALES = (1, 100, 1e-3, 1e-6)
ATTEMPTS = 2
NUMBER_CHARS = "0123456789."
SYSTEM = """You write Hugging Face model cards in the style of livekit/turn-detector: concise, neutral, technical, short
paragraphs, bold key terms, lists for procedures and constraints, no hype, no em dashes. Use only the facts given.
Never write a number that is not in the facts. Structure, with these exact headings:
# <title>, a two-sentence summary and the headline result in one sentence
## Table of Contents (links to the sections below)
## Overview (the problem, why a small specialist, when to use it)
## How It Works (numbered: input, prediction, decision threshold, output)
## Architecture and Training (### Base Model, ### Training Data with the dataset names and row counts, ### Pretraining
and Refinement, ### Quantization and Export)
## Benchmarks (one paragraph on what is compared and how, then the line {{RESULTS}}, then two sentences reading the
results honestly, including where a rival is close or ahead, then the line {{TEST_SETS}})
## Usage (one sentence, then the line {{USAGE}})
## Deployment Requirements (bullets: CPU only, size, latency, dependencies)
## Limitations (bullets from the facts)
## License (one line)
## Resources (bullets: BYOSM on GitHub, the datasets and rivals with links)
## Reproduce (the line {{REPRODUCE}})
Each {{...}} placeholder must appear exactly once on its own line."""


def _rounded(v: Any) -> Any:
    if isinstance(v, float):
        return round(v, 3) if abs(v) < 10 else round(v, 1)
    if isinstance(v, dict):
        return {k: _rounded(x) for k, x in v.items()}
    if isinstance(v, list):
        return [_rounded(x) for x in v]
    return v


def _export(f: CardFacts) -> dict[str, Any]:
    out = f.p.root / "export"
    req = out / "requirements.txt"
    m = f.manifest
    return {
        "format": "ONNX, single folder with a one-file Python runtime",
        "precision": m.get("precision"),
        "size_mb": round(sum(x.stat().st_size for x in out.glob("*.onnx")) / 1e6, 1),
        "decision_threshold": m.get("decision", {}).get("threshold"),
        "max_tokens": m.get("frontend", {}).get("max_len"),
        "requirements": [x.strip() for x in req.read_text().splitlines() if x.strip()] if req.exists() else [],
    }


def facts(f: CardFacts) -> dict[str, Any]:
    r = f.results.get(f.prim, {})
    m = r.get("metrics", {})
    rivals = [
        {**model_ref(f.p, pid, by.get(f.prim)), "score": by.get(f.prim, {}).get("objective")}
        for pid, by in f.board.items()
        if by.get(f.prim, {}).get("kind") == "baseline"
    ]
    return {
        "title": f.base.title,
        "repo": f.repo_id,
        "task": f.base.summary or f.base.description,
        "inputs": f.base.input_description,
        "labels": {lab.name: lab.definition for lab in f.base.labels},
        "languages": f.base.languages,
        "metric": metric_label(f.base.objective),
        "benchmark": suite_dataset(f.base, f.prim)[0],
        "this_model": {
            "experiment": f.exp,
            "score": r.get("objective"),
            "ci95": m.get("ci95"),
            "precision": m.get("precision"),
            "recall": m.get("recall"),
            "test_rows": r.get("n_test"),
            "params_millions": round(f.manifest.get("params", 0) / 1e6, 1),
            "cpu_ms": r.get("latency_ms"),
            "backbone": f.spec.backbone,
        },
        "rivals": rivals,
        "training": {
            "rows": f.meta.get("n_train"),
            "dev_rows": f.meta.get("n_dev"),
            "sources": f.used,
            "epochs": f.spec.train.epochs,
            "pretrain_steps": f.spec.train.pretrain_steps,
            "refinement_rounds": [
                {"rows_before": r.get("n_train"), "dev_score": r.get("dev_objective")} for r in f.meta.get("rounds", [])
            ],
            "refinement_patterns": [
                p.get("name") for a in f.meta.get("refine_analyses", []) for p in a.get("patterns", [])
            ],
            "lineage": f.lineage,
        },
        "export": _export(f),
        "limitations": sections.caveats(f),
        "license": f.license,
        "github": "https://github.com/bsye/BYOSM",
    }


def _numbers(text: str) -> list[float]:
    out, cur = [], ""
    for i, c in enumerate(text + " "):
        if c in NUMBER_CHARS or (c == "," and cur and i + 1 < len(text) and text[i + 1].isdigit()):
            cur += "" if c == "," else c
            continue
        token = cur.strip(".")
        if token and token.count(".") <= 1:
            out.append(float(token))
        cur = ""
    return out


def unsupported(prose: str, allowed_text: str) -> list[float]:
    allowed = _numbers(allowed_text)

    def known(v: float) -> bool:
        return (v.is_integer() and v <= FREE_INTEGERS) or any(
            abs(v - a * k) <= TOLERANCE * max(1.0, abs(a * k)) for a in allowed for k in SCALES
        )

    return [v for v in _numbers(prose) if not known(v)]


def _fill(prose: str, b: dict[str, str]) -> str | None:
    if any(prose.count("{{" + p + "}}") != 1 for p in PLACES):
        return None
    for place, key in PLACES.items():
        prose = prose.replace("{{" + place + "}}", _nested(b[key]))
    return prose


def _nested(block: str) -> str:
    lines = block.strip().splitlines()
    if lines and lines[0].startswith("## ") and lines[0][3:].strip().lower() in SAME_AS_SECTION:
        lines = lines[1:]
    return "\n".join("#" + ln if ln.startswith("##") else ln for ln in lines).strip()


def written_body(f: CardFacts, b: dict[str, str]) -> str | None:
    from ..llm import get_llm, role_model

    given = json.dumps(_rounded(facts(f)), indent=1, default=str)
    allowed_text = given + "".join(b.values())
    prompt, note = f"Facts:\n{given}", ""
    for _ in range(ATTEMPTS):
        try:
            prose = get_llm(CARD_ROLE).json(SYSTEM, prompt + note, SCHEMA).get("markdown", "")
        except Exception:
            return None
        wrong = unsupported(prose.replace("{{", "").replace("}}", ""), allowed_text)
        filled = _fill(prose, b)
        if filled and not wrong:
            credit = f"\n\n---\n\n*Prose by {role_model(CARD_ROLE)} from the facts above; every number is checked against them.*\n"
            return filled + "\n\n" + b["llm"] + credit
        note = f"\n\nYour last draft used numbers not in the facts: {wrong[:10]}. Use only numbers from the facts."
    return None


SCHEMA = {"type": "object", "properties": {"markdown": {"type": "string"}}}
