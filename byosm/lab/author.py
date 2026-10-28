from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from ..project import Project
from .contract import CONTRACT
from .drafts import drafts, save_draft, update

ROLE = "lab"
ERROR_CHARS = 800
FENCE = "```"
SYSTEM = f"""You are a machine-learning researcher. Write one new training method for a small fine-tuned classifier,
aimed at the weakness the diagnosis and the failing health checks show, and different from what was tried. Return a short name, the hypothesis it
tests, and the code of the module.

{CONTRACT}"""
SCHEMA = {
    "type": "object",
    "properties": {"name": {"type": "string"}, "hypothesis": {"type": "string"}, "code": {"type": "string"}},
}


def _earlier(p: Project) -> list[dict[str, Any]]:
    keep = ("name", "hypothesis", "status", "problems", "result")
    return [{**{k: d.get(k) for k in keep}, "error": (d.get("error") or "")[-ERROR_CHARS:]} for d in drafts(p)]


def _paper(paper: str, log: Callable[[str], None]) -> list[str]:
    from ..scout.fulltext import paper_method

    text = paper_method(paper, log) if paper else ""
    return (
        [f"Method sections of the paper to implement ({paper}); follow it where the contract allows:\n{text}"]
        if text
        else []
    )


def _prompt(p: Project, instruction: str, paper: str = "", log: Callable[[str], None] = print) -> str:
    from ..improve.diagnosis import diagnose

    dg = diagnose(p.name)
    suites = {s: {k: r.get(k) for k in ("objective", "weakest", "confusions")} for s, r in dg["suites"].items()}
    tried = [{k: h.get(k) for k in ("experiment", "family", "changes", "calib", "outcome")} for h in dg["history"]]
    parts = [
        f"Task: {dg['spec'].description}\nLabels: {', '.join(dg['spec'].label_names)}",
        f"Promoted model {dg['experiment']} per suite (test objective, weakest labels, confusions):\n{json.dumps(suites)}",
        f"Dev curve per epoch: {dg['curve']}",
        f"Health of the promoted model; the method should clear what is not ok:\n{json.dumps(dg.get('health', []))}",
        f"Experiments tried so far:\n{json.dumps(tried, default=str)}",
        f"Methods written before, with their outcome or error; fix or avoid what failed:\n{json.dumps(_earlier(p))}",
    ]
    parts += _paper(paper, log)
    return "\n\n".join(parts + ([f"The owner asks: {instruction}"] if instruction else []))


def unfenced(code: str) -> str:
    body = code.strip()
    if body.startswith(FENCE):
        body = body.partition("\n")[2]
    return body.rpartition(FENCE)[0].strip() if body.endswith(FENCE) else body


def write(
    name: str, instruction: str = "", log: Callable[[str], None] = print, paper: str = "", parent: str | None = None
) -> dict[str, Any]:
    from ..llm import get_llm, role_model

    p = Project(name)
    out = get_llm(ROLE).json(SYSTEM, _prompt(p, instruction, paper, log), SCHEMA)
    code, title = unfenced(out.get("code") or ""), out.get("name") or "method"
    draft = save_draft(p, title, out.get("hypothesis") or "", code, role_model(ROLE), parent)
    if paper:
        draft = update(p, draft["id"], paper=paper)
    state = "fails its checks: " + "; ".join(draft["problems"]) if draft["problems"] else "waits for approval"
    log(f"[lab] {draft['id']} {draft['name']}: {draft['hypothesis']} ({state})")
    return draft
