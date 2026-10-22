from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from .base import LLMError, _parse_json, _retry, _strict

CODEX_BINARY = "codex"
CODEX_TIMEOUT_S = 600
EMPTY_DIR, SCHEMA_FILE, ANSWER_FILE = "work", "schema.json", "answer.txt"
CODEX_ISOLATION_FLAGS = (
    "--sandbox",
    "read-only",
    "--skip-git-repo-check",
    "--ephemeral",
    "--ignore-user-config",
    "--ignore-rules",
    "-c",
    'approval_policy="never"',
    "-c",
    'web_search="disabled"',
)


class CodexLLM:
    def __init__(
        self,
        model: str | None = None,
        binary: str = CODEX_BINARY,
        timeout_s: int = CODEX_TIMEOUT_S,
        effort: str | None = None,
        **_: Any,
    ):
        self.model, self.binary, self.timeout_s, self.effort = model, binary, timeout_s, effort
        self.name = f"codex:{model or 'default'}"

    def _command(self, system: str, tmp: Path) -> list[str]:
        cmd = [self.binary, "exec", "-C", str(tmp / EMPTY_DIR), *CODEX_ISOLATION_FLAGS]
        cmd += ["--output-schema", str(tmp / SCHEMA_FILE), "-o", str(tmp / ANSWER_FILE)]
        cmd += ["-c", f"developer_instructions={json.dumps(system)}"]
        cmd += ["-c", f"model_reasoning_effort={self.effort}"] if self.effort else []
        model = ["-m", self.model] if self.model else []
        return [*cmd, *model, "-"]

    def _run(self, system: str, prompt: str, schema: dict) -> dict:
        with tempfile.TemporaryDirectory() as tmp_name:
            tmp = Path(tmp_name)
            (tmp / EMPTY_DIR).mkdir()
            (tmp / SCHEMA_FILE).write_text(json.dumps(_strict(schema)))
            done = subprocess.run(
                self._command(system, tmp),
                input=prompt,
                cwd=tmp / EMPTY_DIR,
                capture_output=True,
                text=True,
                timeout=self.timeout_s,
            )
            answer = tmp / ANSWER_FILE
            text = answer.read_text() if answer.exists() else ""
        if done.returncode or not text.strip():
            raise LLMError(f"codex exec failed (exit {done.returncode}): {done.stderr[-300:]}")
        return _parse_json(text)

    def json(self, system: str, prompt: str, schema: dict, images: list[bytes] | None = None) -> dict:
        if images:
            raise LLMError("the codex provider takes text only; use a vision model for image or document roles")
        return _retry(lambda: self._run(system, prompt, schema))
