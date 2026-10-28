from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from ..project import Project
from .contract import CONTRACT
from .drafts import drafts, save_draft

ROLE = "lab"
ERROR_CHARS = 800
FENCE = "```"
SYSTEM = f"""You are a machine-learning researcher. Write one new training method for a small fine-tuned classifier,
aimed at the weakness the diagnosis shows and different from what was tried. Return a short name, the hypothesis it
tests, and the code of the module.

{CONTRACT}"""
SCHEMA = {
    "type": "object",
    "properties": {"name": {"type": "string"}, "hypothesis": {"type": "string"}, "code": {"type": "string"}},
}


def _earlier(p: Project) -> list[dict[str, Any]]:
    keep = ("name", "hypothesis", "status", "problems", "result")
    return [{**{k: d.get(k) for k in keep}, "error": (d.get("error") or "")[-ERROR_CHARS:]} for d in drafts(p)]


def _prompt(p: Project, instruction: str) -> str:
    from ..improve.diagnosis import diagnose

    dg = diagnose(p.name)
    suites = {s: {k: r.get(k) for k in ("objective", "weakest", "confusions")} for s, r in dg["suites"].items()}
    tried = [{k: h.get(k) for k in ("experiment", "family", "changes", "calib", "outcome")} for h in dg["history"]]
    parts = [
        f"Task: {dg['spec'].description}\nLabels: {', '.join(dg['spec'].label_names)}",
        f"Promoted model {dg['experiment']} per suite (test objective, weakest labels, confusions):\n{json.dumps(suites)}",
        f"Dev curve per epoch: {dg['curve']}",
        f"Experiments tried so far:\n{json.dumps(tried, default=str)}",
        f"Methods written before, with their outcome or error; fix or avoid what failed:\n{json.dumps(_earlier(p))}",
    ]
    return "\n\n".join(parts + ([f"The owner asks: {instruction}"] if instruction else []))


def unfenced(code: str) -> str:
    body = code.strip()
    if body.startswith(FENCE):
        body = body.partition("\n")[2]
    return body.rpartition(FENCE)[0].strip() if body.endswith(FENCE) else body


def write(name: str, instruction: str = "", log: Callable[[str], None] = print) -> dict[str, Any]:
    from ..llm import get_llm, role_model

    p = Project(name)
    out = get_llm(ROLE).json(SYSTEM, _prompt(p, instruction), SCHEMA)
    draft = save_draft(
        p, out.get("name") or "method", out.get("hypothesis") or "", unfenced(out.get("code") or ""), role_model(ROLE)
    )
    state = "fails its checks: " + "; ".join(draft["problems"]) if draft["problems"] else "waits for approval"
    log(f"[lab] {draft['id']} {draft['name']}: {draft['hypothesis']} ({state})")
    return draft
