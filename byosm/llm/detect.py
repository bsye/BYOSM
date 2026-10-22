from __future__ import annotations

import json
import shutil
import subprocess
import urllib.request
from collections.abc import Callable
from typing import Any

from ..config import DEFAULT_CONFIG, secret
from . import pmap
from .claude_code_provider import CLAUDE_BINARY, CLAUDE_MODELS
from .codex_provider import CODEX_BINARY

OLLAMA_URL = "http://localhost:11434"
LMSTUDIO_URL = "http://localhost:1234/v1"
OPENAI_URL = "https://api.openai.com/v1"
ANTHROPIC_URL = "https://api.anthropic.com/v1"
ANTHROPIC_VERSION = "2023-06-01"
OPENAI_KEY_ENV, ANTHROPIC_KEY_ENV = "OPENAI_API_KEY", "ANTHROPIC_API_KEY"
OPENAI_CHAT_PREFIX = "gpt-"
OLLAMA_KEY, LMSTUDIO_KEY = "ollama", "lm-studio"
PROBE_TIMEOUT_S = 3

Found = dict[str, Any] | None


def _get_json(url: str, headers: dict[str, str] | None = None) -> dict | None:
    try:
        request = urllib.request.Request(url, headers=headers or {})
        with urllib.request.urlopen(request, timeout=PROBE_TIMEOUT_S) as r:
            return json.loads(r.read())
    except (OSError, ValueError):
        return None


def _ids(listing: dict | None) -> list[str]:
    return [m["id"] for m in (listing or {}).get("data", [])]


def _found(models: list[str], **config: Any) -> dict[str, Any]:
    return {"models": models, "config": {**config, "model": models[0] if models else None}}


def _server(base_url: str, api_key: str, listing: dict | None, models: list[str]) -> Found:
    if listing is None:
        return None
    return _found(models, provider="openai", base_url=base_url, api_key=api_key)


def _ollama() -> Found:
    listing = _get_json(f"{OLLAMA_URL}/api/tags")
    models = [m["name"] for m in (listing or {}).get("models", [])]
    return _server(f"{OLLAMA_URL}/v1", OLLAMA_KEY, listing, models)


def _lmstudio() -> Found:
    listing = _get_json(f"{LMSTUDIO_URL}/models")
    return _server(LMSTUDIO_URL, LMSTUDIO_KEY, listing, _ids(listing))


def _cli(binary: str) -> dict[str, str] | None:
    path = shutil.which(binary)
    if not path:
        return None
    try:
        done = subprocess.run([path, "--version"], capture_output=True, text=True, timeout=PROBE_TIMEOUT_S)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return {"path": path, "version": done.stdout.strip()}


def _claude_code() -> Found:
    cli = _cli(CLAUDE_BINARY)
    return cli and {**cli, **_found(list(CLAUDE_MODELS), provider="claude-code")}


def _codex() -> Found:
    cli = _cli(CODEX_BINARY)
    return cli and {**cli, **_found([], provider="codex")}


def _openai() -> Found:
    key = secret(OPENAI_KEY_ENV)
    if not key:
        return None
    listing = _get_json(f"{OPENAI_URL}/models", {"Authorization": f"Bearer {key}"})
    models = sorted((m for m in _ids(listing) if m.startswith(OPENAI_CHAT_PREFIX)), reverse=True)
    return _found(models, provider="openai", api_key_env=OPENAI_KEY_ENV)


def _anthropic() -> Found:
    key = secret(ANTHROPIC_KEY_ENV)
    if not key:
        return None
    headers = {"x-api-key": key, "anthropic-version": ANTHROPIC_VERSION}
    models = _ids(_get_json(f"{ANTHROPIC_URL}/models", headers)) or [DEFAULT_CONFIG["llm"]["default"]["model"]]
    return _found(models, provider="anthropic", api_key_env=ANTHROPIC_KEY_ENV)


KNOWN: dict[str, tuple[str, str, str, Callable[[], Found]]] = {
    "claude-code": (
        "Claude Code",
        "local agent",
        "Install with npm i -g @anthropic-ai/claude-code, then run claude once",
        _claude_code,
    ),
    "codex": ("Codex", "local agent", "Install with npm i -g @openai/codex, then run codex login", _codex),
    "ollama": ("Ollama", "local model", "Install from ollama.com, then ollama pull a model", _ollama),
    "lmstudio": (
        "LM Studio",
        "local model",
        "Install from lmstudio.ai, load a model and start its local server",
        _lmstudio,
    ),
    "openai": ("OpenAI API", "hosted", f"Set {OPENAI_KEY_ENV} in the environment or .env", _openai),
    "anthropic": ("Anthropic API", "hosted", f"Set {ANTHROPIC_KEY_ENV} in the environment or .env", _anthropic),
}


def _entry(pid: str, found: Found) -> dict[str, Any]:
    label, kind, hint, _ = KNOWN[pid]
    base = {"id": pid, "label": label, "kind": kind, "available": found is not None, "models": [], "config": None}
    return {**base, **found} if found else {**base, "hint": hint}


def detect() -> dict[str, Any]:
    found = pmap(lambda known: known[3](), KNOWN.values(), concurrency=len(KNOWN))
    return {"providers": [_entry(pid, f) for pid, f in zip(KNOWN, found, strict=True)]}
