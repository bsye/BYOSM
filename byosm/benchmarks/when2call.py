from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

from ..data.store import read_file
from ..options.situations import Situation
from ..spec import TaskSpec
from .pairs import Case, option, pair_rows
from .source import download

REPO = "nvidia/When2Call"
TEST_FILE = "test/when2call_test_mcq.jsonl"
TRAIN_FILES = ("train/when2call_train_pref.jsonl", "train/when2call_train_sft.jsonl")
CALL, NO_CALL = "tool_call", "no_call"
TOOLCALL_OPEN, TOOLCALL_CLOSE = "<TOOLCALL>", "</TOOLCALL>"


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


def called(reply: str) -> set[str]:
    body = reply.strip()
    if not body.startswith(TOOLCALL_OPEN):
        return set()
    return {c["name"] for c in json.loads(body.removeprefix(TOOLCALL_OPEN).partition(TOOLCALL_CLOSE)[0])}


def train_cases() -> Iterator[Case]:
    for file in TRAIN_FILES:
        for i, r in enumerate(read_file(fetch(file))):
            offered = [tool(t) for t in r["tools"]]
            reply = (r.get("chosen_response") or r["messages"][-1])["content"]
            question = next(m["content"] for m in r["messages"] if m["role"] == "user")
            chosen = called(reply)
            if offered:
                yield Case(f"{file}:{i}", CALL if chosen else NO_CALL, Situation(str(question).strip(), offered, chosen))


def rows(spec: TaskSpec, log: Callable[[str], None]) -> list[dict]:
    return pair_rows(spec, cases(), REPO)
