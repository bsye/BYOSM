from __future__ import annotations

import json
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from .anthropic_provider import AnthropicLLM
from .base import LLM, LLMError, T
from .openai_provider import OpenAICompatLLM

__all__ = [
    "LLM",
    "PROVIDERS",
    "AnthropicLLM",
    "LLMError",
    "OpenAICompatLLM",
    "_cache",
    "get_llm",
    "pmap",
    "register_provider",
]

PROVIDERS: dict[str, Callable[..., LLM]] = {"anthropic": AnthropicLLM, "openai": OpenAICompatLLM}


def register_provider(name: str, factory: Callable[..., LLM]) -> None:
    PROVIDERS[name] = factory


_cache: dict[str, LLM] = {}


def get_llm(role: str = "default", cfg: dict | None = None) -> LLM:
    from ..config import load_config

    cfg = cfg or load_config()
    roles = cfg["llm"]
    conf = dict(roles.get(role) or roles["default"])
    key = json.dumps(conf, sort_keys=True)
    if key not in _cache:
        provider = conf.pop("provider", "anthropic")
        if provider not in PROVIDERS:
            raise LLMError(f"unknown provider {provider!r}; known: {sorted(PROVIDERS)}")
        _cache[key] = PROVIDERS[provider](**conf)
    return _cache[key]


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
