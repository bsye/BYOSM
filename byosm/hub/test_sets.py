from __future__ import annotations

from collections import Counter
from typing import Any

from ..evals.test_sets import test_sets
from ..evals.workbench import OURS
from ..model_refs import model_ref
from .data_release import data_repo
from .facts import CardFacts

LLM_SOURCE = "llm:"
HEAD = """
## Results on test sets written by an LLM

Inputs an LLM wrote and labelled for this task, scored by the same code. They are not a public benchmark, and an LLM
rival from the same family as the writer may have an edge. Every input, label and answer is in
[{data}](https://huggingface.co/datasets/{data}).
"""


def _sets(f: CardFacts) -> list[dict[str, Any]]:
    return [s for s in test_sets(f.p, f.base) if not s["board"] and s["rows"]]


def _answers(f: CardFacts, s: dict, model: str) -> dict[str, dict]:
    ids = {r["id"] for r in s["rows"]}
    got = s["results"].get(model, {})
    return {i: a for i, a in got.items() if i in ids and (model != OURS or a.get("exp") in (None, f.exp))}


def _name(f: CardFacts, model: str) -> str:
    return f"**{f.exp} (this model)**" if model == OURS else model_ref(f.p, model)["name"]


def _writers(s: dict) -> str:
    found = Counter(str(r.get("source", "")).removeprefix(LLM_SOURCE) for r in s["rows"])
    return ", ".join(f"`{w}`" for w, _ in found.most_common()) or "an LLM"


def _table(f: CardFacts, s: dict) -> list[str]:
    labels = {r["id"]: r["label"] for r in s["rows"]}
    lines = [f"### `{s['suite']}`: {len(labels):,} inputs written by {_writers(s)}", "", "| model | right | accuracy |"]
    lines.append("|---|---|---|")
    scored = []
    for model in s["results"]:
        got = _answers(f, s, model)
        right = sum(a["label"] == labels.get(i) for i, a in got.items())
        if got:
            scored.append(
                (right / len(got), f"| {_name(f, model)} | {right:,} / {len(got):,} | {right / len(got):.3f} |")
            )
    return lines + [row for _, row in sorted(scored, reverse=True)] + [""]


def section(f: CardFacts) -> str:
    sets = _sets(f)
    if not sets:
        return ""
    return HEAD.format(data=data_repo(f.repo_id)) + "\n" + "\n".join(line for s in sets for line in _table(f, s))
