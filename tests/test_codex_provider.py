import json
import subprocess
from pathlib import Path

import pytest

from byosm.llm import PROVIDERS, CodexLLM, LLMError

SCHEMA = {"type": "object", "properties": {"label": {"type": "string"}}}


def _flag(cmd: list[str], flag: str) -> str:
    return cmd[cmd.index(flag) + 1]


def _fake_codex(answer: str | None, returncode: int = 0, seen: dict | None = None):
    def run(cmd, input, cwd, **_):
        if seen is not None:
            seen.update(cmd=cmd, input=input, schema=json.loads(Path(_flag(cmd, "--output-schema")).read_text()))
            seen["workdir"] = list(Path(_flag(cmd, "-C")).iterdir())
        if answer is not None:
            Path(_flag(cmd, "-o")).write_text(answer)
        return subprocess.CompletedProcess(cmd, returncode, stdout="", stderr="boom")

    return run


def test_codex_is_registered():
    assert PROVIDERS["codex"] is CodexLLM


def test_codex_runs_exec_isolated_with_a_strict_schema(monkeypatch):
    seen: dict = {}
    monkeypatch.setattr(subprocess, "run", _fake_codex('{"label": "spam"}', seen=seen))
    out = CodexLLM(model="gpt-test").json("be brief", "classify this", SCHEMA)
    cmd = seen["cmd"]
    assert out == {"label": "spam"}
    assert cmd[:2] == ["codex", "exec"] and cmd[-1] == "-"
    assert seen["input"] == "classify this"
    assert _flag(cmd, "--sandbox") == "read-only"
    assert {"--skip-git-repo-check", "--ephemeral", "--ignore-user-config", "--ignore-rules"} <= set(cmd)
    assert {'approval_policy="never"', 'web_search="disabled"', 'developer_instructions="be brief"'} <= set(cmd)
    assert _flag(cmd, "-m") == "gpt-test"
    assert seen["schema"]["additionalProperties"] is False
    assert seen["workdir"] == []


def test_codex_without_a_model_uses_the_cli_default(monkeypatch):
    seen: dict = {}
    monkeypatch.setattr(subprocess, "run", _fake_codex('```json\n{"label": "ham"}\n```', seen=seen))
    llm = CodexLLM()
    assert llm.json("s", "p", SCHEMA) == {"label": "ham"}
    assert "-m" not in seen["cmd"] and llm.name == "codex:default"


def test_codex_failure_raises(monkeypatch):
    monkeypatch.setattr(subprocess, "run", _fake_codex(None, returncode=1))
    with pytest.raises(LLMError, match="exit 1"):
        CodexLLM()._run("s", "p", SCHEMA)


def test_codex_takes_text_only():
    with pytest.raises(LLMError, match="text only"):
        CodexLLM().json("s", "p", SCHEMA, images=[b"img"])
