from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from ..project import Project
from .contract import CONTRACTS
from .drafts import drafts, save_draft, update
from .owners import Forecaster, Owner, kind_of, owner

ROLE = "lab"
ERROR_CHARS = 800
FENCE = "```"
EARLIER = "Methods written before, with their outcome or error; fix or avoid what failed:"
HEALTH = "Health of the promoted model; the method should clear what is not ok:"
STEPS = "Next steps the system lines up on its own:"
FORECASTING = """Forecasting domain {name}: one model per recipe is trained on all of its datasets ({datasets}) at once.
Scores are MASE and CRPS on held-out windows, lower is better; the board divides each by seasonal naive's, a version
is picked and promoted on the calib windows, and test is only reported."""
SYSTEM = """You are a machine-learning researcher. Write one new training method for {model}, aimed at {goal}, and
different from what was tried. Return a short name, the hypothesis it tests, and the code of the module.

{contract}"""
SCHEMA = {
    "type": "object",
    "properties": {"name": {"type": "string"}, "hypothesis": {"type": "string"}, "code": {"type": "string"}},
}


def system(kind: str) -> str:
    c = CONTRACTS[kind]
    return SYSTEM.format(model=c.model, goal=c.goal, contract=c.text)


def _earlier(found: Owner) -> list[dict[str, Any]]:
    keep = ("name", "hypothesis", "status", "problems", "result")
    return [{**{k: d.get(k) for k in keep}, "error": (d.get("error") or "")[-ERROR_CHARS:]} for d in drafts(found)]


def _paper(paper: str, log: Callable[[str], None]) -> list[str]:
    from ..scout.fulltext import paper_method

    text = paper_method(paper, log) if paper else ""
    return (
        [f"Method sections of the paper to implement ({paper}); follow it where the contract allows:\n{text}"]
        if text
        else []
    )


def _classifier_facts(p: Project) -> list[str]:
    from ..improve.diagnosis import diagnose

    dg = diagnose(p.name)
    suites = {s: {k: r.get(k) for k in ("objective", "weakest", "confusions")} for s, r in dg["suites"].items()}
    tried = [{k: h.get(k) for k in ("experiment", "family", "changes", "calib", "outcome")} for h in dg["history"]]
    return [
        f"Task: {dg['spec'].description}\nKind: {dg['spec'].modality} {dg['spec'].task_type}\nLabels: {', '.join(dg['spec'].label_names)}",
        f"Promoted model {dg['experiment']} per suite (test objective, weakest labels, confusions):\n{json.dumps(suites)}",
        f"Dev curve per epoch: {dg['curve']}",
        *_loop(dg),
        f"Experiments tried so far:\n{json.dumps(tried, default=str)}",
    ]


def _loop(dg: dict[str, Any]) -> list[str]:
    return [f"{HEALTH}\n{json.dumps(dg.get('health', []))}", f"{STEPS}\n{json.dumps(dg.get('next_steps', []))}"]


def _forecaster_facts(f: Forecaster) -> list[str]:
    from ..series.diagnosis import diagnose

    dg = diagnose(f.ws, f.name)
    return [
        FORECASTING.format(name=f.name, datasets=", ".join(dg["datasets"])),
        f"Promoted version and its options:\n{json.dumps(dg['promoted'])}",
        f"Best rival on the datasets both were scored on, both relative to seasonal naive:\n{json.dumps(dg['rival'])}",
        f"Scores per dataset of every model the promoted version was scored with:\n{json.dumps(dg['board'])}",
        *_loop(dg),
        f"Versions trained so far:\n{json.dumps(dg['versions'])}",
    ]


def _prompt(found: Owner, instruction: str, paper: str = "", log: Callable[[str], None] = print) -> str:
    facts = _forecaster_facts(found) if isinstance(found, Forecaster) else _classifier_facts(found)
    parts = [*facts, f"{EARLIER}\n{json.dumps(_earlier(found))}", *_paper(paper, log)]
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

    found = owner(name)
    out = get_llm(ROLE).json(system(kind_of(found)), _prompt(found, instruction, paper, log), SCHEMA)
    code, title = unfenced(out.get("code") or ""), out.get("name") or "method"
    draft = save_draft(found, title, out.get("hypothesis") or "", code, role_model(ROLE), parent)
    if paper:
        draft = update(found, draft["id"], paper=paper)
    state = "fails its checks: " + "; ".join(draft["problems"]) if draft["problems"] else "waits for approval"
    log(f"[lab] {draft['id']} {draft['name']}: {draft['hypothesis']} ({state})")
    return draft
