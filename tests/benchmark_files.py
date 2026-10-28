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


def word(text, start, end):
    return {"start": start, "end": end, "word": text}


HELLO = [word("hello", 0.0, 0.3), word("there", 0.4, 0.8), word("how", 1.3, 1.5), word("are", 1.6, 1.8)]
EOT_TURNS = [
    {
        "id": "en_a",
        "silence_spans": [{"start": 0.8, "end": 1.3}, {"start": 1.5, "end": 1.58}, {"start": 2.2, "end": 3.4}],
        "words": [*HELLO, word("you", 1.9, 2.2)],
    },
    {
        "id": "en_f",
        "silence_spans": [{"start": 0.5, "end": 0.75}, {"start": 1.0, "end": 1.15}, {"start": 1.6, "end": 2.6}],
        "words": [word("yes", 0.1, 0.5), word("please", 0.75, 1.0), word("now", 1.15, 1.6)],
    },
    {
        "id": "en_b",
        "silence_spans": [{"start": 0.4, "end": 1.1}, {"start": 1.5, "end": 2.0}],
        "words": [word("my", 0.0, 0.2), word("number", 0.2, 0.4), word("is", 1.1, 1.5)],
    },
]
EOT_HOLDS = (0.5, 0.25, 0.7)


def eot_summary(language, at_300, at_600, model="VAD baseline"):
    budgets = {"best_cutoff_rate_at_0_3s_latency": at_300, "best_cutoff_rate_at_0_6s_latency": at_600}
    return {"values": {k: {language: {model: v, "LiveKit Turn Detector v1": 0.05}} for k, v in budgets.items()}}
