from __future__ import annotations

import json
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from .anthropic_provider import AnthropicLLM
from .base import LLM, LLMError, T
from .claude_code_provider import ClaudeCodeLLM
from .codex_provider import CodexLLM
from .openai_provider import OpenAICompatLLM

__all__ = [
    "LLM",
    "PROVIDERS",
    "AnthropicLLM",
    "ClaudeCodeLLM",
    "CodexLLM",
    "LLMError",
    "OpenAICompatLLM",
    "_cache",
    "get_llm",
    "pmap",
    "register_provider",
]

PROVIDERS: dict[str, Callable[..., LLM]] = {
    "anthropic": AnthropicLLM,
    "openai": OpenAICompatLLM,
    "claude-code": ClaudeCodeLLM,
    "codex": CodexLLM,
}


def register_provider(name: str, factory: Callable[..., LLM]) -> None:
    PROVIDERS[name] = factory


_cache: dict[str, LLM] = {}


CONNECTION = ("provider", "base_url", "api_key", "api_key_env")
PREVIEW_CHARS = 600


def role_config(roles: dict, role: str) -> dict:
    default = roles["default"]
    own = roles.get(role) or {}
    if any(k in own for k in CONNECTION):
        return {"provider": default.get("provider", "anthropic"), **own}
    return {**default, **own}


def get_llm(role: str = "default", cfg: dict | None = None) -> LLM:
    from ..config import load_config

    cfg = cfg or load_config()
    conf = role_config(cfg["llm"], role)
    key = json.dumps(conf, sort_keys=True)
    if key not in _cache:
        provider = conf.pop("provider", "anthropic")
        if provider not in PROVIDERS:
            raise LLMError(f"unknown provider {provider!r}; known: {sorted(PROVIDERS)}")
        _cache[key] = Observed(PROVIDERS[provider](**conf), role)
    return _cache[key]


class Observed:
    def __init__(self, llm: LLM, role: str):
        self.llm, self.role = llm, role
        self.name = getattr(llm, "name", type(llm).__name__)

    def __getattr__(self, attr: str) -> Any:
        return getattr(self.llm, attr)

    def json(self, system: str, prompt: str, schema: dict, images: list[bytes] | None = None) -> dict:
        import time
        import uuid

        from ..events import emit

        t0, call = time.time(), uuid.uuid4().hex[:8]
        base = {"call": call, "role": self.role, "model": self.name, "at": round(t0, 2)}
        emit("llm", **base, state="waiting", prompt=prompt[:PREVIEW_CHARS], images=len(images or []))
        try:
            out = self.llm.json(system, prompt, schema, **({"images": images} if images else {}))
        except Exception as e:
            emit("llm", **base, state="failed", seconds=round(time.time() - t0, 1), error=str(e)[:PREVIEW_CHARS])
            raise
        answer = json.dumps(out, ensure_ascii=False)[:PREVIEW_CHARS]
        emit("llm", **base, state="done", seconds=round(time.time() - t0, 1), answer=answer)
        return out


def pmap(
    fn: Callable[[Any], T],
    items: Iterable[Any],
    concurrency: int = 8,
    on_error: Callable[[Exception], None] | None = None,
) -> list[T]:
    items = list(items)
    out: list[T] = []
    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as ex:
        futures = [ex.submit(fn, it) for it in items]
        for f in futures:
            try:
                out.append(f.result())
            except Exception as e:
                if on_error:
                    on_error(e)
                else:
                    raise
    return out
