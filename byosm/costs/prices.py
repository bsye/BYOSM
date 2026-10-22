from __future__ import annotations

import json
import time
import urllib.request
from functools import lru_cache
from typing import Any

PRICES_URL = "https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json"
PRICES_FILE = "llm-prices.json"
REFRESH_S = 24 * 3600
TIMEOUT_S = 20
LOCAL_MARKERS = ("localhost", "127.0.0.1", "ollama", "lm-studio")
PROVIDER_PREFIXES = ("claude-code:", "anthropic:", "openai:", "codex:")


def _table() -> dict[str, Any]:
    return _loaded(int(time.time() // REFRESH_S))


@lru_cache(maxsize=1)
def _loaded(_day: int) -> dict[str, Any]:
    from ..config import cache_root

    path = cache_root() / PRICES_FILE
    if not path.exists() or time.time() - path.stat().st_mtime > REFRESH_S:
        try:
            with urllib.request.urlopen(PRICES_URL, timeout=TIMEOUT_S) as r:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(r.read())
        except OSError:
            if not path.exists():
                return {}
    return json.loads(path.read_text())


def is_local(model: str) -> bool:
    return any(m in model for m in LOCAL_MARKERS)


def bare(model: str) -> str:
    name = model.partition("@")[0]
    return next((name.removeprefix(p) for p in PROVIDER_PREFIXES if name.startswith(p)), name)


def price(model: str) -> tuple[float, float] | None:
    if is_local(model):
        return (0.0, 0.0)
    row = _table().get(bare(model))
    if not row or row.get("input_cost_per_token") is None:
        return None
    return float(row["input_cost_per_token"]), float(row.get("output_cost_per_token") or 0.0)
