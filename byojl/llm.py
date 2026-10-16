"""Pluggable LLM layer. Every call in byojl is `llm.json(system, prompt, schema) -> dict`.

Providers:
  anthropic  Claude through the official `anthropic` SDK (structured outputs).
  openai     Any OpenAI-compatible server (OpenAI, Ollama, vLLM, LM Studio, OpenRouter, Together...).

Roles (see byojl.yaml): spec, generate, judge, eval, analyze -> each falls back to `default`.
"""
from __future__ import annotations

import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Iterable, Protocol, TypeVar

T = TypeVar("T")


class LLMError(RuntimeError):
    pass


class LLM(Protocol):
    name: str

    def json(self, system: str, prompt: str, schema: dict, images: list[bytes] | None = None) -> dict:
        """images: PNG/JPEG bytes shown before the prompt (multimodal models only)."""
        ...


def _strict(schema: dict) -> dict:
    """Make a JSON schema acceptable to strict structured-output modes: every object closed, every key required."""
    s = dict(schema)
    if s.get("type") == "object":
        props = {k: _strict(v) for k, v in s.get("properties", {}).items()}
        s["properties"] = props
        s["required"] = list(props)
        s["additionalProperties"] = False
    if s.get("type") == "array" and "items" in s:
        s["items"] = _strict(s["items"])
    return s


def _parse_json(text: str) -> dict:
    text = text.strip()
    m = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if m:
        text = m.group(1)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start >= 0 and end > start:
            return json.loads(text[start : end + 1])
        raise


def _retry(fn: Callable[[], T], tries: int = 4) -> T:
    for i in range(tries):
        try:
            return fn()
        except (json.JSONDecodeError, LLMError):
            if i == tries - 1:
                raise
        except Exception as e:  # network / rate limit: back off
            if i == tries - 1:
                raise
            if "rate" not in str(e).lower() and "overload" not in str(e).lower() and "timeout" not in str(e).lower() and "connect" not in str(e).lower():
                raise
            time.sleep(2 ** (i + 1))
    raise AssertionError("unreachable")


def _b64(img: bytes) -> tuple[str, str]:
    import base64

    return ("image/png" if img[:4] == b"\x89PNG" else "image/jpeg"), base64.b64encode(img).decode()


class AnthropicLLM:
    # Models that accept server-side refusal fallbacks (`fallbacks: "default"`).
    _FALLBACK_PREFIXES = ("claude-opus-5", "claude-fable-5", "claude-sonnet-5-5")

    def __init__(self, model: str = "claude-opus-5-5", effort: str | None = "medium", max_tokens: int = 16000, api_key_env: str | None = None, base_url: str | None = None, **_: Any):
        import anthropic

        kwargs: dict[str, Any] = {}
        if api_key_env:
            kwargs["api_key"] = os.environ[api_key_env]
        if base_url:
            kwargs["base_url"] = base_url
        self.client = anthropic.Anthropic(**kwargs)
        self.model, self.effort, self.max_tokens = model, effort, max_tokens
        self.name = f"anthropic:{model}"

    def json(self, system: str, prompt: str, schema: dict, images: list[bytes] | None = None) -> dict:
        output_config: dict[str, Any] = {"format": {"type": "json_schema", "schema": _strict(schema)}}
        content: Any = prompt
        if images:
            content = [{"type": "image", "source": {"type": "base64", "media_type": mt, "data": d}} for mt, d in map(_b64, images)] + [{"type": "text", "text": prompt}]
        if self.effort:
            output_config["effort"] = self.effort
        kwargs: dict[str, Any] = dict(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system,
            messages=[{"role": "user", "content": content}],
            output_config=output_config,
        )

        def call() -> dict:
            if self.model.startswith(self._FALLBACK_PREFIXES):
                resp = self.client.beta.messages.create(betas=["server-side-fallback-2026-07-01"], fallbacks="default", **kwargs)
            else:
                resp = self.client.messages.create(**kwargs)
            if resp.stop_reason == "refusal":
                raise LLMError(f"{self.name} refused: {getattr(resp, 'stop_details', None)}")
            if resp.stop_reason == "max_tokens":
                raise LLMError(f"{self.name} hit max_tokens={self.max_tokens}; lower batch size")
            text = "".join(b.text for b in resp.content if b.type == "text")
            return _parse_json(text)

        return _retry(call)


class OpenAICompatLLM:
    def __init__(self, model: str, base_url: str | None = None, api_key: str | None = None, api_key_env: str | None = None, temperature: float | None = None, max_tokens: int = 8000, **_: Any):
        import openai

        key = api_key or (os.environ.get(api_key_env) if api_key_env else None) or os.environ.get("OPENAI_API_KEY") or "not-needed"
        self.client = openai.OpenAI(base_url=base_url, api_key=key)
        self.model, self.temperature, self.max_tokens = model, temperature, max_tokens
        self.name = f"openai:{model}@{base_url or 'api.openai.com'}"
        self._schema_mode = True  # fall back to json_object if the server rejects json_schema

    def json(self, system: str, prompt: str, schema: dict, images: list[bytes] | None = None) -> dict:
        def call() -> dict:
            user: Any = prompt
            if images:
                user = [{"type": "image_url", "image_url": {"url": f"data:{mt};base64,{d}"}} for mt, d in map(_b64, images)] + [{"type": "text", "text": prompt}]
            messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
            kwargs: dict[str, Any] = dict(model=self.model, messages=messages, max_tokens=self.max_tokens)
            if self.temperature is not None:
                kwargs["temperature"] = self.temperature
            if self._schema_mode:
                kwargs["response_format"] = {"type": "json_schema", "json_schema": {"name": "output", "schema": _strict(schema), "strict": True}}
            else:
                suffix = f"\n\nRespond with only a JSON object matching this JSON schema:\n{json.dumps(schema)}"
                if isinstance(messages[1]["content"], list):
                    messages[1]["content"][-1]["text"] += suffix
                else:
                    messages[1]["content"] = prompt + suffix
                kwargs["response_format"] = {"type": "json_object"}
            try:
                resp = self.client.chat.completions.create(**kwargs)
            except Exception as e:
                if self._schema_mode and ("response_format" in str(e) or "json_schema" in str(e)):
                    self._schema_mode = False
                    return call()
                raise
            return _parse_json(resp.choices[0].message.content or "")

        return _retry(call)


PROVIDERS: dict[str, Callable[..., LLM]] = {"anthropic": AnthropicLLM, "openai": OpenAICompatLLM}


def register_provider(name: str, factory: Callable[..., LLM]) -> None:
    """Plug in any other backend (e.g. a LangGraph node, a Bedrock client, a local llama.cpp binding)."""
    PROVIDERS[name] = factory


_cache: dict[str, LLM] = {}


def get_llm(role: str = "default", cfg: dict | None = None) -> LLM:
    from .config import load_config

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


def pmap(fn: Callable[[Any], T], items: Iterable[Any], concurrency: int = 8, on_error: Callable[[Exception], None] | None = None) -> list[T]:
    """Run LLM calls concurrently; failed items are reported and skipped, never silently retried forever."""
    items = list(items)
    out: list[T] = []
    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as ex:
        futures = [ex.submit(fn, it) for it in items]
        for f in futures:
            try:
                out.append(f.result())
            except Exception as e:  # noqa: BLE001
                if on_error:
                    on_error(e)
                else:
                    raise
    return out
