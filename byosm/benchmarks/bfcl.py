from __future__ import annotations

from collections.abc import Callable, Iterator
from pathlib import Path

from ..data.store import read_file
from ..options.situations import Situation
from ..spec import TaskSpec
from .pairs import Case, option, pair_rows
from .source import download

REPO = "gorilla-llm/Berkeley-Function-Calling-Leaderboard"
VERSION = "v3"
NON_LIVE = ("simple", "multiple", "irrelevance")
LIVE = ("live_simple", "live_multiple", "live_irrelevance")
NO_TOOL = ("irrelevance", "live_irrelevance")
QUESTIONS = "BFCL_{version}_{category}.json"
ANSWERS = "possible_answer/BFCL_{version}_{category}.json"
USER = "user"


def fetch(file: str) -> Path:
    return download(REPO, file)


def request(question: list[list[dict]]) -> str:
    return "\n".join(str(m["content"]).strip() for m in question[0] if m["role"] == USER)


def called(answer: dict) -> set[str]:
    return {next(iter(call)) for call in answer["ground_truth"]}


def answers(category: str) -> dict[str, set[str]]:
    if category in NO_TOOL:
        return {}
    return {a["id"]: called(a) for a in read_file(fetch(ANSWERS.format(version=VERSION, category=category)))}


def cases(categories: tuple[str, ...]) -> Iterator[Case]:
    for category in categories:
        chosen_by_id = answers(category)
        for q in read_file(fetch(QUESTIONS.format(version=VERSION, category=category))):
            offered = [option(t) for t in q["function"]]
            chosen = chosen_by_id.get(q["id"], set())
            fair = category in NO_TOOL or bool(chosen)
            if offered and fair and chosen <= {o["name"] for o in offered}:
                yield Case(q["id"], category, Situation(request(q["question"]), offered, chosen))


def rows(categories: tuple[str, ...], spec: TaskSpec, log: Callable[[str], None]) -> list[dict]:
    return pair_rows(spec, cases(categories), REPO)
