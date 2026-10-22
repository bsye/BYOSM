from __future__ import annotations

import contextlib
import json
import threading
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from .anthropic_provider import AnthropicLLM
from .base import LLM, LLMError, T, take_usage
from .claude_code_provider import ClaudeCodeLLM
from .codex_provider import CodexLLM
from .openai_provider import OpenAICompatLLM
from .recover import through_limits

__all__ = [
    "LLM",
    "PROVIDERS",
    "AnthropicLLM",
    "ClaudeCodeLLM",
    "CodexLLM",
    "LLMError",
    "OpenAICompatLLM",
    "clear_cache",
    "get_llm",
    "pmap",
    "register_provider",
    "role_model",
]

PROVIDERS: dict[str, Callable[..., LLM]] = {
    "anthropic": AnthropicLLM,
    "openai": OpenAICompatLLM,
    "claude-code": ClaudeCodeLLM,
    "codex": CodexLLM,
}


def clear_cache() -> None:
    _cache.clear()
    _providers.clear()


def register_provider(name: str, factory: Callable[..., LLM]) -> None:
    PROVIDERS[name] = factory


_cache: dict[tuple[str, str], LLM] = {}
_spent = {"tokens": 0, "budget": 0}
BUDGET_KEY = "tokens_per_job"
_providers: dict[str, LLM] = {}
_gates: dict[str, threading.BoundedSemaphore] = {}
LOCAL_HOSTS = ("localhost", "127.0.0.1", "[::1]")


CONNECTION = ("provider", "base_url", "api_key", "api_key_env")
PREVIEW_CHARS = 600


def role_config(roles: dict, role: str) -> dict:
    default = roles["default"]
    own = roles.get(role) or {}
    if any(k in own for k in CONNECTION):
        return {"provider": default.get("provider", "anthropic"), **own}
    return {**default, **own}


def role_model(role: str, cfg: dict | None = None) -> str:
    from ..config import load_config

    conf = role_config((cfg or load_config())["llm"], role)
    return conf.get("model") or conf.get("provider", "anthropic")


def get_llm(role: str = "default", cfg: dict | None = None) -> LLM:
    from ..config import load_config

    cfg = cfg or load_config()
    _spent["budget"] = int((cfg.get("budget") or {}).get(BUDGET_KEY) or 0)
    conf = role_config(cfg["llm"], role)
    key = json.dumps(conf, sort_keys=True)
    if (role, key) not in _cache:
        provider = conf.pop("provider", "anthropic")
        if provider not in PROVIDERS:
            raise LLMError(f"unknown provider {provider!r}; known: {sorted(PROVIDERS)}")
        limit = _limit(conf)
        shared = _providers.get(key) or _providers.setdefault(key, PROVIDERS[provider](**conf))
        _cache[role, key] = Observed(shared, role, _gate(conf, limit), limit)
    return _cache[role, key]


def _limit(conf: dict) -> int | None:
    from ..resources import local_parallel

    local = any(host in str(conf.get("base_url") or "") for host in LOCAL_HOSTS)
    limit = conf.pop("max_parallel", None) or (local_parallel() if local else None)
    return int(limit) if limit else None


def _gate(conf: dict, limit: int | None) -> threading.BoundedSemaphore | None:
    if not limit:
        return None
    endpoint = f"{conf.get('base_url') or ''}|{conf.get('model')}"
    return _gates.setdefault(endpoint, threading.BoundedSemaphore(limit))


class Observed:
    def __init__(self, llm: LLM, role: str, gate: threading.BoundedSemaphore | None = None, limit: int | None = None):
        self.llm, self.role, self.gate, self.limit = llm, role, gate, limit
        self.name = getattr(llm, "name", type(llm).__name__)

    def parallel(self, concurrency: int) -> int:
        return max(1, min(concurrency, self.limit or concurrency))

    def __getattr__(self, attr: str) -> Any:
        return getattr(self.llm, attr)

    def json(self, system: str, prompt: str, schema: dict, images: list[bytes] | None = None) -> dict:
        import time
        import uuid

        from ..events import emit

        t0, call = time.time(), uuid.uuid4().hex[:8]
        base = {"call": call, "role": self.role, "model": self.name, "at": round(t0, 2)}
        if _spent["budget"] and _spent["tokens"] >= _spent["budget"]:
            raise LLMError(f"token budget reached: {_spent['tokens']} of {_spent['budget']} tokens for this job")
        emit("llm", **base, state="waiting", prompt=prompt[:PREVIEW_CHARS], images=len(images or []))
        t1 = t0
        try:
            out, t1 = through_limits(lambda llm: self._once(llm, system, prompt, schema, images), self.llm, base)
        except Exception as e:
            timing = {"seconds": round(time.time() - t1, 1), "waited": round(t1 - t0, 1)}
            emit("llm", **base, state="failed", **timing, error=str(e)[:PREVIEW_CHARS])
            raise
        answer = json.dumps(out, ensure_ascii=False)[:PREVIEW_CHARS]
        used = take_usage()
        _spent["tokens"] += (used["tokens_in"] or 0) + (used["tokens_out"] or 0)
        timing = {"seconds": round(time.time() - t1, 1), "waited": round(t1 - t0, 1)}
        emit("llm", **base, state="done", **timing, answer=answer, **used)
        return out

    def _once(self, llm: LLM, system: str, prompt: str, schema: dict, images: list[bytes] | None) -> tuple[dict, float]:
        import time

        with self.gate or contextlib.nullcontext():
            t1 = time.time()
            return llm.json(system, prompt, schema, **({"images": images} if images else {})), t1


def parallel_calls(cfg: dict | None = None) -> int:
    from ..config import load_config
    from ..resources import REMOTE_PARALLEL

    return (cfg if cfg is not None else load_config()).get("concurrency", REMOTE_PARALLEL)


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
