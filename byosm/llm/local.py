from __future__ import annotations

import json
import shutil
import subprocess
import urllib.request
from typing import Any

from .detect import OLLAMA_URL, PROBE_TIMEOUT_S, _get_json

OLLAMA, LMSTUDIO = "ollama", "lmstudio"
LMS_BINARY = "lms"
CLI_TIMEOUT_S = 20


def _ollama() -> list[dict[str, Any]]:
    listing = _get_json(f"{OLLAMA_URL}/api/ps") or {}
    return [
        {
            "server": OLLAMA,
            "model": m["name"],
            "bytes": m.get("size_vram") or m.get("size"),
            "params": (m.get("details") or {}).get("parameter_size"),
            "quantization": (m.get("details") or {}).get("quantization_level"),
            "until": m.get("expires_at"),
        }
        for m in listing.get("models", [])
    ]


def _lmstudio() -> list[dict[str, Any]]:
    lms = shutil.which(LMS_BINARY)
    if not lms:
        return []
    done = subprocess.run([lms, "ps", "--json"], capture_output=True, text=True, timeout=CLI_TIMEOUT_S)
    try:
        rows = json.loads(done.stdout or "[]")
    except json.JSONDecodeError:
        return []
    return [
        {
            "server": LMSTUDIO,
            "model": r.get("identifier") or r.get("modelKey") or r.get("path"),
            "bytes": r.get("sizeBytes"),
            "params": r.get("paramsString"),
            "quantization": (r.get("quantization") or {}).get("name")
            if isinstance(r.get("quantization"), dict)
            else None,
            "until": None,
        }
        for r in rows
    ]


def loaded() -> list[dict[str, Any]]:
    return [*_ollama(), *_lmstudio()]


def unload(server: str, model: str) -> None:
    if server == OLLAMA:
        body = json.dumps({"model": model, "keep_alive": 0}).encode()
        req = urllib.request.Request(
            f"{OLLAMA_URL}/api/generate", data=body, headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=PROBE_TIMEOUT_S * 10):
            return
    if server == LMSTUDIO and (lms := shutil.which(LMS_BINARY)):
        subprocess.run([lms, "unload", model], check=True, capture_output=True, timeout=CLI_TIMEOUT_S)
        return
    raise ValueError(f"cannot unload from {server!r}: known servers are {OLLAMA} and {LMSTUDIO}")
