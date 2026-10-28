from __future__ import annotations

from collections.abc import Callable

from ..data.synth_text import role
from ..llm import get_llm, parallel_calls, role_model
from ..llm.memo import memo_key, memoised

WRITER_ROLE = role("train")
REWRITE_BATCH = 20
REWRITE_VERSION = "1"
REWRITE_SYSTEM = """You rewrite requests that people send to an assistant that calls tools. Keep every rewrite natural, \
in the request's language, and follow the instruction exactly."""
PARAPHRASE = """Write {k} rewrites of each request below. Change the wording, order and tone; keep its meaning and every \
value it gives (names, numbers, dates, quoted text) unchanged."""
LEAVE_OUT = """Write 1 rewrite of each request below that no longer gives the value named in brackets after it and keeps \
everything else. Leave the brackets out of the rewrite."""
SCHEMA = {
    "type": "object",
    "properties": {
        "rewrites": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"n": {"type": "integer"}, "texts": {"type": "array", "items": {"type": "string"}}},
            },
        }
    },
}

Log = Callable[[str], None]


def flat(text: str) -> str:
    return " ".join(str(text).split())


def writer(cfg: dict | None) -> str:
    return role_model(WRITER_ROLE, cfg)


def rewrites(items: list[str], instruction: str, cfg: dict | None, log: Log) -> list[list[str]]:
    llm = get_llm(WRITER_ROLE, cfg)
    keys = [memo_key(REWRITE_VERSION, writer(cfg), instruction, flat(it)) for it in items]
    text_of = dict(zip(keys, items, strict=True))

    def compute(part: list[str]) -> dict[str, list[str]]:
        listing = "\n".join(f"{i}. {flat(text_of[k])}" for i, k in enumerate(part, 1))
        out = llm.json(REWRITE_SYSTEM, f"{instruction}\n\n{listing}", SCHEMA).get("rewrites") or []
        got = {r.get("n"): [flat(t) for t in r.get("texts") or [] if flat(t)] for r in out if isinstance(r, dict)}
        return {k: got[i] for i, k in enumerate(part, 1) if got.get(i)}

    return [found or [] for found in memoised(keys, compute, REWRITE_BATCH, parallel_calls(cfg), log)]


def paraphrases(requests: list[str], k: int, cfg: dict | None, log: Log) -> list[list[str]]:
    return [texts[:k] for texts in rewrites(requests, PARAPHRASE.format(k=k), cfg, log)]


def left_out(items: list[tuple[str, str]], cfg: dict | None, log: Log) -> list[str | None]:
    found = rewrites([f"{request} [{what}]" for request, what in items], LEAVE_OUT, cfg, log)
    return [texts[0] if texts else None for texts in found]
