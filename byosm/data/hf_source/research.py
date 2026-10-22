from __future__ import annotations

import shutil
from collections.abc import Callable
from typing import Any

from ...llm import get_llm, role_model
from ...llm.claude_code_provider import CLAUDE_BINARY, CLAUDE_MODEL
from ...spec import TaskSpec
from .hub import exists

RESEARCH_ROLE = "research"
WEB_TOOLS = "WebSearch,WebFetch"
MAX_FOUND = 8
RESEARCH_SYSTEM = """You find public Hugging Face datasets to train a small classifier. Search the web and huggingface.co. \
Return only datasets you checked exist at https://huggingface.co/datasets/<owner>/<name>, as their exact owner/name id, \
each with one line on why it fits: what its labels are, how many rows, its licence. Prefer labelled data whose labels \
match the task; include unlabelled in-domain text only if nothing labelled exists."""


def research_config(cfg: dict) -> dict[str, Any] | None:
    own = (cfg.get("llm") or {}).get(RESEARCH_ROLE)
    if own:
        return own
    return (
        {"provider": "claude-code", "model": CLAUDE_MODEL, "tools": WEB_TOOLS} if shutil.which(CLAUDE_BINARY) else None
    )


def research(spec: TaskSpec, cfg: dict, log: Callable[[str], None]) -> list[dict]:
    conf = research_config(cfg)
    if not conf:
        return []
    with_research = {**cfg, "llm": {**cfg["llm"], RESEARCH_ROLE: conf}}
    log(f"  searching the web for datasets with {role_model(RESEARCH_ROLE, with_research)}")
    item = {"type": "object", "properties": {"id": {"type": "string"}, "why": {"type": "string"}}}
    schema = {"type": "object", "properties": {"datasets": {"type": "array", "items": item}}}
    task = f"Task: {spec.summary or spec.description}\nInputs: {spec.input_description}\nLabels: {spec.label_names}"
    out = get_llm(RESEARCH_ROLE, with_research).json(RESEARCH_SYSTEM, task, schema)
    found = [d for d in out.get("datasets", []) if isinstance(d, dict) and d.get("id")][:MAX_FOUND]
    real = [d for d in found if exists(d["id"])]
    log(f"  the web search named {len(found)} datasets, {len(real)} exist on the Hub")
    return [{"id": d["id"], "downloads": 0, "tags": [], "why": d.get("why", "")} for d in real]
