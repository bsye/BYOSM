from __future__ import annotations

import json
import shutil
import subprocess
import urllib.request
from typing import Any

from .claude_code_provider import CLAUDE_BINARY, CLAUDE_MODEL

OLLAMA_URL = "http://localhost:11434"
PROBE_TIMEOUT_S = 3


def _ollama() -> dict[str, Any] | None:
    try:
        with urllib.request.urlopen(f"{OLLAMA_URL}/api/tags", timeout=PROBE_TIMEOUT_S) as r:
            models = [m["name"] for m in json.loads(r.read()).get("models", [])]
    except OSError:
        return None
    return {
        "kind": "local model",
        "provider": "openai",
        "base_url": f"{OLLAMA_URL}/v1",
        "api_key": "ollama",
        "models": models,
    }


def _claude_code() -> dict[str, Any] | None:
    path = shutil.which(CLAUDE_BINARY)
    if not path:
        return None
    try:
        version = subprocess.run([path, "--version"], capture_output=True, text=True, timeout=PROBE_TIMEOUT_S)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return {
        "kind": "local agent",
        "provider": "claude-code",
        "path": path,
        "version": version.stdout.strip(),
        "models": [CLAUDE_MODEL],
    }


def detect() -> dict[str, Any]:
    return {name: found for name, found in (("ollama", _ollama()), ("claude-code", _claude_code())) if found}
