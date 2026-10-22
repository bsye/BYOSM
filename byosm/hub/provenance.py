from __future__ import annotations

import json

from ..format import fmt
from .facts import CardFacts

DEFINE_STEP = "define"
EXAMPLE_CHARS = 120
LLM_NOTE = (
    "Everything in this section was written by **{by}** from the one-sentence task above, when the spec was compiled. "
    "It shapes what the model learns but was not checked by hand; the numbers above were computed by code."
)
SPEC_NOTE = "From `spec.yaml`, shipped in this repo."
NO_RIVAL = "No rival scored yet."


def spec_author(f: CardFacts) -> str | None:
    auto = f.p.root / "auto.json"
    if not auto.exists():
        return None
    first = next((d for d in json.loads(auto.read_text()).get("decisions", []) if d.get("step") == DEFINE_STEP), None)
    return (first.get("by") or "an LLM") if first else None


def headline(f: CardFacts) -> str:
    from ..auto.standing import best_baseline

    mine = f.results.get(f.prim, {})
    ci = (mine.get("metrics") or {}).get("ci95") or {}
    score = f"**{fmt(mine.get('objective'), 3)}** test {f.base.objective.metric}"
    interval = f" [{fmt(ci['lo'], 3)}, {fmt(ci['hi'], 3)}]" if ci else ""
    rival = best_baseline(f.board, f.prim)
    theirs = f.board.get(rival[0], {}).get(f.prim, {}).get("objective") if rival else None
    ours = mine.get("objective")
    if theirs is None or ours is None:
        return f"{score}{interval} on `{f.prim}`. {NO_RIVAL}"
    return f"{score}{interval} on `{f.prim}`, {ours - theirs:+.3f} over `{rival[0]}` ({fmt(theirs, 3)})"


def llm_section(f: CardFacts) -> str:
    by = spec_author(f)
    spec = f.spec
    title = "Written by an LLM" if by else "Task definition"
    note = LLM_NOTE.format(by=by) if by else SPEC_NOTE
    rows = [f"## {title}", "", note, ""]
    if spec.summary:
        rows += [f"**Task.** {spec.summary}", ""]
    if spec.input_description:
        rows += [f"**Inputs.** {spec.input_description}", ""]
    rows += ["**Labels.**", ""] + [f"- `{lab.name}`: {lab.definition}" for lab in spec.labels]
    if spec.hard_cases:
        rows += ["", "**Hard cases the data was asked to cover.**", ""] + [f"- {h}" for h in spec.hard_cases]
    return "\n".join(rows) + "\n"


def example_input(f: CardFacts) -> str | None:
    from ..data.store import Dataset

    if f.spec.modality != "text" or f.spec.task_type != "classify":
        return None
    rows = Dataset(f.p.data_dir_for(f.spec)).load("test")
    return str(rows[0]["input"])[:EXAMPLE_CHARS] if rows else None
