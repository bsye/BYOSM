from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .project import Project

HUB_URL = "https://huggingface.co/"
HUB_DIR = "hub"
HUB_SEPARATOR = "__"
PRODUCT = "byosm"
MANIFEST = "manifest.json"
LLM_PREFIX = "llm"
LOCAL_LLM_HOSTS = ("localhost", "127.0.0.1")
LLM_PAGES = {
    "anthropic": "https://docs.anthropic.com/en/docs/about-claude/models",
    "claude-code": "https://docs.anthropic.com/en/docs/about-claude/models",
    "openai": "https://platform.openai.com/docs/models",
}
OLLAMA_LIBRARY = "https://ollama.com/library/"


def published(p: Project) -> dict[str, str | None]:
    hub = p.root / HUB_DIR
    dirs = [d for d in hub.iterdir() if d.is_dir()] if hub.exists() else []
    found = {d.name.replace(HUB_SEPARATOR, "/", 1): (_json(d / MANIFEST) or {}).get("experiment") for d in dirs}
    return dict(sorted(found.items(), key=lambda kv: PRODUCT not in kv[0]))


def published_repos(p: Project) -> list[str]:
    return list(published(p))


def _json(f: Path) -> Any:
    return json.loads(f.read_text()) if f.exists() else None


def _llm(pid: str, identity: str | None) -> dict[str, Any]:
    from .evals.identity import baseline_identity

    conf = json.loads(identity or baseline_identity(pid) or "{}")
    model, provider = conf.get("model", "?"), conf.get("provider", "?")
    local = any(h in str(conf.get("base_url", "")) for h in LOCAL_LLM_HOSTS)
    url = f"{OLLAMA_LIBRARY}{model.partition(':')[0]}" if local else LLM_PAGES.get(provider)
    effort = conf.get("effort") or conf.get("reasoning_effort")
    detail = f"{model} via {'Ollama' if local else provider}" + (f", effort {effort}" if effort else "")
    return {"name": "the LLM" if pid == LLM_PREFIX else model, "detail": detail, "url": url}


def _baseline(pid: str) -> dict[str, Any]:
    from .predictors import BASELINES

    if pid.startswith("hf:"):
        repo = pid[3:].partition("#")[0]
        return {"name": repo, "detail": "Hugging Face model", "url": HUB_URL + repo}
    cls = BASELINES.get(pid)
    repo = getattr(cls, "hub_repo", None)
    home = HUB_URL + repo if repo else getattr(cls, "home", None)
    return {"name": pid, "detail": getattr(cls, "summary", ""), "url": home}


def model_ref(p: Project, pid: str, result: dict | None = None) -> dict[str, Any]:
    if pid == LLM_PREFIX or pid.startswith(f"{LLM_PREFIX}:"):
        return {"id": pid, **_llm(pid, (result or {}).get("identity"))}
    if pid in p.experiments():
        repos = [repo for repo, exp in published(p).items() if exp == pid]
        mine = "this model" if pid == p.promoted else "an experiment"
        detail = mine + (f", published as {repos[0]}" if repos else ", not published")
        return {"id": pid, "name": pid, "detail": detail, "url": HUB_URL + repos[0] if repos else None}
    return {"id": pid, **_baseline(pid)}


def model_refs(p: Project, board: dict[str, dict[str, dict]]) -> dict[str, dict[str, Any]]:
    scored = {pid: model_ref(p, pid, next(iter(by.values()), None)) for pid, by in board.items()}
    return {**{b: model_ref(p, b) for b in p.spec().baselines}, **scored}
