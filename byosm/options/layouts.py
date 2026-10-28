from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

DESCRIPTION_CHARS = 300
PARAM_CHARS = 80
OPTION_MARK = "\n\nOption: "
OTHERS_MARK = "\nOther options: "
NO_OTHERS = "none"
NO_PARAMS = "no parameters"
OPTIONAL = "optional"
MASKED_NAME = "tool_{}"

Option = dict[str, Any]


def params(option: Option) -> dict[str, dict]:
    return {k: v if isinstance(v, dict) else {} for k, v in (option.get("parameters") or {}).items()}


def required(option: Option) -> list[str]:
    return [
        k
        for k, v in params(option).items()
        if v.get("required", "default" not in v and OPTIONAL not in str(v.get("type") or ""))
    ]


def description(option: Option) -> str:
    return str(option.get("description") or "")[:DESCRIPTION_CHARS]


def option_text(option: Option) -> str:
    return f"{option['name']}: {description(option)} ({', '.join(params(option))})"


def other_names(option: Option, offered: list[Option]) -> list[str]:
    return [o["name"] for o in offered if o["name"] != option["name"]]


def schema(option: Option) -> dict[str, Any]:
    props = {
        k: {"type": str(v.get("type") or "string"), "description": str(v.get("description") or "")[:PARAM_CHARS]}
        for k, v in params(option).items()
    }
    shape = {"type": "object", "properties": props, "required": required(option)}
    return {"name": option["name"], "description": description(option), "parameters": shape}


def marked(request: str, option: Option, offered: list[Option]) -> str:
    return f"{request}{OPTION_MARK}{option_text(option)}{OTHERS_MARK}{', '.join(other_names(option, offered)) or NO_OTHERS}"


def json_tool(request: str, option: Option, offered: list[Option]) -> str:
    return f"{request}\n\nTool: {json.dumps(schema(option))}\nOther tools: {json.dumps(other_names(option, offered))}"


def openai(request: str, option: Option, offered: list[Option]) -> str:
    payload = {
        "messages": [{"role": "user", "content": request}],
        "tools": [{"type": "function", "function": schema(option)}],
        "other_tools": other_names(option, offered),
    }
    return json.dumps(payload)


def prose(request: str, option: Option, offered: list[Option]) -> str:
    takes = ", ".join(params(option)) or NO_PARAMS
    others = ", ".join(other_names(option, offered)) or NO_OTHERS
    return f"Offered tool: {option['name']}. {description(option)} It takes {takes}. Also offered: {others}.\nRequest: {request}"


RENDERERS: dict[str, Callable[[str, Option, list[Option]], str]] = {
    "marked": marked,
    "json": json_tool,
    "openai": openai,
    "prose": prose,
}


def render(layout: str, request: str, option: Option, offered: list[Option]) -> str:
    return RENDERERS[layout](request, option, offered)


def masked(option: Option, offered: list[Option]) -> tuple[Option, list[Option]]:
    order = [option["name"], *other_names(option, offered)]
    names = {name: MASKED_NAME.format(i) for i, name in enumerate(order, 1)}
    return {**option, "name": names[option["name"]]}, [{**o, "name": names[o["name"]]} for o in offered]
