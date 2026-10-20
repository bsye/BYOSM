from __future__ import annotations

import os
from typing import Any

from .base import LLMError, _b64, _parse_json, _retry, _strict


class AnthropicLLM:
    _FALLBACK_PREFIXES = ("claude-opus-5", "claude-fable-5", "claude-sonnet-5-5")

    def __init__(
        self,
        model: str = "claude-opus-5-5",
        effort: str | None = "medium",
        max_tokens: int = 16000,
        api_key_env: str | None = None,
        base_url: str | None = None,
        **_: Any,
    ):
        import anthropic

        kwargs: dict[str, Any] = {}
        if api_key_env:
            kwargs["api_key"] = os.environ[api_key_env]
        if base_url:
            kwargs["base_url"] = base_url
        self.client = anthropic.Anthropic(**kwargs)
        self.model, self.effort, self.max_tokens = model, effort, max_tokens
        self.name = f"anthropic:{model}"

    def _request(self, system: str, prompt: str, schema: dict, images: list[bytes] | None) -> dict[str, Any]:
        output_config: dict[str, Any] = {"format": {"type": "json_schema", "schema": _strict(schema)}}
        content: Any = prompt
        if images:
            content = [
                {"type": "image", "source": {"type": "base64", "media_type": mt, "data": d}}
                for mt, d in map(_b64, images)
            ] + [{"type": "text", "text": prompt}]
        if self.effort:
            output_config["effort"] = self.effort
        return dict(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system,
            messages=[{"role": "user", "content": content}],
            output_config=output_config,
        )

    def _call(self, kwargs: dict[str, Any]) -> dict:
        if self.model.startswith(self._FALLBACK_PREFIXES):
            resp = self.client.beta.messages.create(
                betas=["server-side-fallback-2026-07-01"], fallbacks="default", **kwargs
            )
        else:
            resp = self.client.messages.create(**kwargs)
        if resp.stop_reason == "refusal":
            raise LLMError(f"{self.name} refused: {getattr(resp, 'stop_details', None)}")
        if resp.stop_reason == "max_tokens":
            raise LLMError(f"{self.name} hit max_tokens={self.max_tokens}; lower batch size")
        text = "".join(b.text for b in resp.content if b.type == "text")
        return _parse_json(text)

    def json(self, system: str, prompt: str, schema: dict, images: list[bytes] | None = None) -> dict:
        kwargs = self._request(system, prompt, schema, images)
        return _retry(lambda: self._call(kwargs))
