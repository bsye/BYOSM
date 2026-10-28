from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .db.workspace import save_workspace_report, workspace_report

IDEAS = "ideas"
MAX_IDEAS = 8
TRENDING_SHOWN = 12
SYSTEM = """You find small specialist models worth building. A specialist is a fine-tuned encoder of under 1B parameters \
that classifies or tags text, audio, images or documents, and runs on a CPU. Search the web and huggingface.co for tasks \
where people use a large model or an LLM today but a small specialist could match it: the task needs a public benchmark \
on the Hugging Face Hub with labels, and a rival to beat. Skip tasks the workspace already covers. For each idea give: a \
short title, a one-sentence prompt describing the task for the builder, why it is worth building (the demand you saw), \
the benchmark dataset as its exact owner/name id that you checked exists, and the strongest rival model id."""


def _schema() -> dict[str, Any]:
    keys = ("title", "prompt", "why", "dataset", "rival")
    idea = {"type": "object", "properties": {k: {"type": "string"} for k in keys}}
    return {"type": "object", "properties": {IDEAS: {"type": "array", "items": idea}}}


def _context(ws: Path) -> str:
    from .opportunities import trending
    from .project import Project, list_projects

    have = [f"- {n}: {Project(n, ws).spec().description}" for n in list_projects(ws)]
    hot = [f"- {t['id']} ({t['task']}, {t['downloads']} downloads)" for t in trending()[:TRENDING_SHOWN]]
    return "Already built:\n" + "\n".join(have) + "\n\nTrending on Hugging Face:\n" + "\n".join(hot)


def find_ideas(ws: Path, log: Callable[[str], None] = print) -> list[dict[str, Any]]:
    from .config import load_config
    from .data.hf_source.hub import exists
    from .data.hf_source.research import RESEARCH_ROLE, research_config
    from .llm import get_llm

    cfg = load_config(ws)
    conf = research_config(cfg)
    if not conf:
        raise RuntimeError("no LLM with web search found: install Claude Code or set llm.research in byosm.yaml")
    llm = get_llm(RESEARCH_ROLE, {**cfg, "llm": {**cfg["llm"], RESEARCH_ROLE: conf}})
    log(f"  researching ideas with {llm.name}")
    found = [i for i in llm.json(SYSTEM, _context(ws), _schema()).get(IDEAS, []) if i.get("prompt")][:MAX_IDEAS]
    real = [i for i in found if i.get("dataset") and exists(i["dataset"])]
    log(f"  {len(found)} ideas, {len(real)} with a benchmark that exists on the Hub")
    save(ws, real)
    return real


def save(ws: Path, ideas: list[dict[str, Any]]) -> None:
    save_workspace_report(ws, IDEAS, {"at": time.time(), IDEAS: ideas})


def saved(ws: Path) -> dict[str, Any]:
    return workspace_report(ws, IDEAS) or {"at": None, IDEAS: []}
