from __future__ import annotations

import json
import subprocess
import tempfile
from typing import Any

from .base import LLMError, _parse_json, _retry, report_usage

CLAUDE_BINARY = "claude"
CLAUDE_MODEL = "claude-haiku-5-5"
CLAUDE_MODELS = (CLAUDE_MODEL, "claude-sonnet-5-5", "claude-opus-5-5", "claude-fable-5-1")
CLAUDE_TIMEOUT_S = 600
INPUT_KEYS = ("input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens")
ISOLATION_FLAGS = ("--strict-mcp-config", "--setting-sources", "project", "--disable-slash-commands", "--tools", "")


class ClaudeCodeLLM:
    def __init__(
        self,
        model: str = CLAUDE_MODEL,
        binary: str = CLAUDE_BINARY,
        timeout_s: int = CLAUDE_TIMEOUT_S,
        effort: str | None = None,
        **_: Any,
    ):
        self.model, self.binary, self.timeout_s, self.effort = model, binary, timeout_s, effort
        self.name = f"claude-code:{model}"

    def _run(self, system: str, prompt: str, schema: dict) -> dict:
        cmd = [self.binary, "-p", prompt, "--model", self.model, "--output-format", "json"]
        cmd += ["--json-schema", json.dumps(schema), "--system-prompt", system, "--no-session-persistence"]
        cmd += ["--effort", self.effort] if self.effort else []
        with tempfile.TemporaryDirectory() as empty:
            done = subprocess.run(
                [*cmd, *ISOLATION_FLAGS], cwd=empty, capture_output=True, text=True, timeout=self.timeout_s
            )
        try:
            out = json.loads(done.stdout)
        except json.JSONDecodeError as e:
            raise LLMError(f"claude -p gave no JSON (exit {done.returncode}): {done.stderr[-300:]}") from e
        if out.get("is_error"):
            raise LLMError(f"claude -p failed: {str(out.get('result'))[:300]}")
        used = out.get("usage") or {}
        report_usage(sum(int(used.get(k) or 0) for k in INPUT_KEYS), used.get("output_tokens"))
        return out.get("structured_output") or _parse_json(str(out.get("result", "")))

    def json(self, system: str, prompt: str, schema: dict, images: list[bytes] | None = None) -> dict:
        if images:
            raise LLMError("the claude-code provider takes text only; use a vision model for image or document roles")
        return _retry(lambda: self._run(system, prompt, schema))
