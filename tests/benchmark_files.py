import json

WEATHER = {
    "name": "get_weather",
    "description": "Current weather for a city.",
    "parameters": {
        "type": "dict",
        "properties": {
            "city": {"type": "string", "description": "City name."},
            "unit": {"type": "string", "default": "c"},
        },
        "required": ["city"],
    },
}
STOCK = {
    "name": "get_stock",
    "description": "Latest share price.",
    "parameters": {"type": "dict", "properties": {"ticker": {"type": "string"}}, "required": ["ticker"]},
}


def ask(text, system=None):
    return [[*([{"role": "system", "content": system}] if system else []), {"role": "user", "content": text}]]


BFCL = {
    "BFCL_v3_simple.json": [
        {"id": "simple_0", "question": ask("Weather in Paris?"), "function": [WEATHER]},
        {"id": "simple_1", "question": ask("Price of ACME?"), "function": [WEATHER]},
    ],
    "possible_answer/BFCL_v3_simple.json": [
        {"id": "simple_0", "ground_truth": [{"get_weather": {"city": ["Paris"]}}]},
        {"id": "simple_1", "ground_truth": [{"get_stock": {"ticker": ["ACME"]}}]},
    ],
    "BFCL_v3_multiple.json": [
        {"id": "multiple_0", "question": ask("Price of ACME?", "Be brief."), "function": [WEATHER, STOCK]},
        {"id": "multiple_11", "question": ask("Rain in Oslo?"), "function": [WEATHER, STOCK]},
    ],
    "possible_answer/BFCL_v3_multiple.json": [
        {"id": "multiple_0", "ground_truth": [{"get_stock": {"ticker": ["ACME"]}}]},
        {"id": "multiple_11", "ground_truth": [{"get_weather": {"city": ["Oslo"]}}]},
    ],
    "BFCL_v3_irrelevance.json": [
        {"id": "irrelevance_0", "question": ask("Solve x^2 = 4."), "function": [WEATHER]},
        {"id": "irrelevance_9", "question": ask("Who wrote Hamlet?"), "function": [STOCK]},
    ],
}


def when2call_row(uuid, answer, tools, source_id, target=WEATHER):
    return {
        "uuid": uuid,
        "source": "BFCL v2 Live Multiple",
        "source_id": source_id,
        "question": "What is the weather in Rome?",
        "correct_answer": answer,
        "answers": {},
        "target_tool": json.dumps(target) if answer != "cannot_answer" else None,
        "tools": [json.dumps(t) for t in tools],
        "orig_tools": [json.dumps(t) for t in (WEATHER, STOCK)],
    }


WHEN2CALL = {
    "test/when2call_test_mcq.jsonl": [
        when2call_row("u1", "tool_call", [WEATHER, STOCK], "live_multiple_0-0-0"),
        when2call_row("u2", "cannot_answer", [STOCK], "live_multiple_0-0-0"),
        when2call_row("u3", "request_for_info", [WEATHER, STOCK], "live_multiple_0-0-0"),
        when2call_row("u4", "cannot_answer", [], "live_simple_1-1-0"),
    ]
}


def written(root, files):
    for name, rows in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return lambda file: root / file
