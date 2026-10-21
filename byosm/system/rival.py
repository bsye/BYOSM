from __future__ import annotations

from collections.abc import Callable

FRAME_BATCH = 10
LOG_EVERY_BATCHES = 50
PROMPT_VERSION = "1"
EMPTY_FRAME = {"action": None, "args": []}

SYSTEM_PROMPT = """You turn requests to an assistant into one action and its arguments.
Actions: {actions}
Argument types: {types}
For each numbered request, return its action and every argument in it. An argument is a type and the exact words \
copied from the request. Many requests have no arguments. Return one item per request, in order."""


def _schema(actions: list[str], types: list[str]) -> dict:
    arg = {"type": "object", "properties": {"type": {"type": "string", "enum": types}, "text": {"type": "string"}}}
    item = {
        "type": "object",
        "properties": {"action": {"type": "string", "enum": actions}, "args": {"type": "array", "items": arg}},
    }
    return {"type": "object", "properties": {"items": {"type": "array", "items": item}}}


def _frame(item: dict) -> dict:
    return {"action": item.get("action"), "args": [[a.get("type"), a.get("text", "")] for a in item.get("args", [])]}


def llm_frames(
    texts: list[str], actions: list[str], types: list[str], log: Callable[[str], None], role: str = "baseline"
) -> list[dict]:
    from ..llm import get_llm

    llm = get_llm(role)
    system = SYSTEM_PROMPT.format(actions=", ".join(actions), types=", ".join(types))
    schema = _schema(actions, types)
    out: list[dict] = []
    for i in range(0, len(texts), FRAME_BATCH):
        chunk = texts[i : i + FRAME_BATCH]
        prompt = "\n".join(f"{j + 1}. {t}" for j, t in enumerate(chunk))
        items = llm.json(system, prompt, schema).get("items", [])
        out += [_frame(it) for it in items[: len(chunk)]] + [EMPTY_FRAME] * (len(chunk) - len(items))
        if (i // FRAME_BATCH) % LOG_EVERY_BATCHES == 0:
            log(f"  llm frames {len(out)}/{len(texts)}")
    return out
