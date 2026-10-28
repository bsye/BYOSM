from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

from ..data.store import read_file
from ..options.situations import Situation
from ..spec import TaskSpec
from .pairs import Case, download, option, pair_rows

REPO = "nvidia/When2Call"
TEST_FILE = "test/when2call_test_mcq.jsonl"
CALL = "tool_call"


def fetch(file: str) -> Path:
    return download(REPO, file)


def tool(raw: str | dict[str, Any]) -> dict[str, Any]:
    return option(json.loads(raw) if isinstance(raw, str) else raw)


def cases() -> Iterator[Case]:
    for r in read_file(fetch(TEST_FILE)):
        offered = [tool(t) for t in r["tools"]]
        chosen = {tool(r["target_tool"])["name"]} if r["correct_answer"] == CALL else set()
        if offered:
            situation = Situation(str(r["question"]).strip(), offered, chosen)
            yield Case(r["uuid"], r["correct_answer"], situation, r["source_id"])


def rows(spec: TaskSpec, log: Callable[[str], None]) -> list[dict]:
    return pair_rows(spec, cases(), REPO)
