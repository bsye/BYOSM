from __future__ import annotations

import json
import os
from typing import Any

from .base import _b64, _parse_json, _retry, _strict


class OpenAICompatLLM:
    def __init__(
        self,
        model: str,
        base_url: str | None = None,
        api_key: str | None = None,
        api_key_env: str | None = None,
        temperature: float | None = None,
        max_tokens: int = 8000,
        **_: Any,
    ):
        import openai

        env_key = os.environ.get(api_key_env) if api_key_env else None
        key = api_key or env_key or os.environ.get("OPENAI_API_KEY") or "not-needed"
        self.client = openai.OpenAI(base_url=base_url, api_key=key)
        self.model, self.temperature, self.max_tokens = model, temperature, max_tokens
        self.name = f"openai:{model}@{base_url or 'api.openai.com'}"
        self._schema_mode = True

    def _request(self, system: str, prompt: str, schema: dict, images: list[bytes] | None) -> dict[str, Any]:
        user: Any = prompt
        if images:
            user = [
                {"type": "image_url", "image_url": {"url": f"data:{mt};base64,{d}"}} for mt, d in map(_b64, images)
            ] + [{"type": "text", "text": prompt}]
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        kwargs: dict[str, Any] = dict(model=self.model, messages=messages, max_tokens=self.max_tokens)
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        if self._schema_mode:
            json_schema = {"name": "output", "schema": _strict(schema), "strict": True}
            kwargs["response_format"] = {"type": "json_schema", "json_schema": json_schema}
        else:
            suffix = f"\n\nRespond with only a JSON object matching this JSON schema:\n{json.dumps(schema)}"
            if isinstance(messages[1]["content"], list):
                messages[1]["content"][-1]["text"] += suffix
            else:
                messages[1]["content"] = prompt + suffix
            kwargs["response_format"] = {"type": "json_object"}
        return kwargs

    def _call(self, system: str, prompt: str, schema: dict, images: list[bytes] | None) -> dict:
        kwargs = self._request(system, prompt, schema, images)
        try:
            resp = self.client.chat.completions.create(**kwargs)
        except Exception as e:
            if self._schema_mode and ("response_format" in str(e) or "json_schema" in str(e)):
                self._schema_mode = False
                return self._call(system, prompt, schema, images)
            raise
        return _parse_json(resp.choices[0].message.content or "")

    def json(self, system: str, prompt: str, schema: dict, images: list[bytes] | None = None) -> dict:
        return _retry(lambda: self._call(system, prompt, schema, images))
