from __future__ import annotations

import json
import math
from collections.abc import Callable, Iterator
from typing import Any

from ..data.store import record
from ..evals.cutoffs import AT_S, LATENCY_BUDGETS_MS, SPAN, SPAN_S
from ..hub.facts import HUB_SOURCE
from ..predictors.turn import FINISHED_NAMES, finished_label
from ..spec import TaskSpec
from .protocol import Kind, Published, Values
from .source import split_of

REPO = "livekit/eot-bench-data"
FILE = "data/{language}/validation-00000-of-00001.parquet"
COLUMNS = ["id", "silence_spans", "words"]
LANGUAGES = ("ar", "de", "en", "es", "fr", "hi", "id", "it", "ja", "ko", "nl", "pt", "tr", "zh")
TRANSCRIPT_LAG_S = 0.5
STEP_S = 0.1
MIN_SPAN_S = 0.1
HOLD_S = (0.2, 5.0)
HARNESS_EPS = 1e-6
METRIC_EPS = 1e-9
TURN = "turn"
COMMIT = "b1cc6abceca916d994c4d044aa8285cba7fab9b2"
PUBLISHED = (
    f"https://raw.githubusercontent.com/livekit/eot-bench/{COMMIT}/output/"
    "livekit__eot-bench-data__validation__min_silence_100ms/language_comparison/summary.json"
)
PUBLISHED_KEYS = {300: "best_cutoff_rate_at_0_3s_latency", 600: "best_cutoff_rate_at_0_6s_latency"}
SILENCE_TIMEOUT = "VAD baseline"
CUTOFF_AT = "cutoff at {} ms"


def turns(language: str) -> list[dict[str, Any]]:
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem

    with HfFileSystem().open(f"datasets/{REPO}/{FILE.format(language=language)}", "rb") as f:
        return pq.ParquetFile(f).read(columns=COLUMNS).to_pylist()


def stamps(start: float, end: float) -> list[float]:
    steps = math.floor((end - start) / STEP_S + HARNESS_EPS)
    found = [round(start + i * STEP_S, 6) for i in range(steps + 1)]
    return found + [round(end, 6)] if found[-1] < end - HARNESS_EPS else found


def heard(words: list[dict], at: float) -> str:
    said = (str(w["word"]).strip() for w in words if w["end"] <= at - TRANSCRIPT_LAG_S + HARNESS_EPS)
    return " ".join(w for w in said if w)


def states(words: list[dict], start: float, end: float) -> Iterator[tuple[float, str]]:
    seen = None
    for at in stamps(start, end):
        text = heard(words, at)
        if text != seen:
            seen = text
            yield round(at - start, 6), text


def length(start: float, end: float) -> float:
    return round(stamps(start, end)[-1] - start, 6)


def kept(start: float, end: float, last: bool) -> bool:
    if end - start < MIN_SPAN_S - HARNESS_EPS:
        return False
    return last or HOLD_S[0] - METRIC_EPS <= length(start, end) <= HOLD_S[1] + METRIC_EPS


def turn_rows(turn: dict[str, Any], finished: str, pause: str, source: str) -> list[dict]:
    spans, split, rows = turn["silence_spans"], split_of(turn["id"]), []
    for i, s in enumerate(spans):
        last = i == len(spans) - 1
        if not kept(s["start"], s["end"], last):
            continue
        for k, (at, text) in enumerate(states(turn["words"] or [], s["start"], s["end"])):
            meta = {SPAN: f"{turn['id']}:{i}", SPAN_S: length(s["start"], s["end"]), AT_S: at, TURN: turn["id"]}
            label = finished if last else pause
            rows.append({**record(text, label, source, **meta), "id": f"{turn['id']}:{i}:{k}", "split": split})
    return rows


def labels(spec: TaskSpec) -> tuple[str, str]:
    finished = finished_label(spec)
    return finished, next(n for n in spec.label_names if n != finished)


def rows(spec: TaskSpec, log: Callable[[str], None]) -> list[dict]:
    finished, pause = labels(spec)
    return [r for t in turns(spec.languages[0]) for r in turn_rows(t, finished, pause, HUB_SOURCE + REPO)]


def fits(spec: TaskSpec) -> bool:
    positive = spec.label_names[spec.pos_index]
    return len(spec.labels) == 2 and positive.lower() in FINISHED_NAMES and spec.languages[0] in LANGUAGES


def summary() -> dict[str, Any]:
    from ..opendata.cache import fetch, table

    fetch(PUBLISHED, log=lambda _: None)
    return json.loads(table(PUBLISHED).read_text())


def published(spec: TaskSpec) -> Published:
    values, language = summary()["values"], spec.languages[0]
    models = values[PUBLISHED_KEYS[LATENCY_BUDGETS_MS[0]]].get(language, {})
    by_budget = {CUTOFF_AT.format(b): values[k].get(language, {}) for b, k in PUBLISHED_KEYS.items()}
    return {m: {at: found.get(m) for at, found in by_budget.items()} for m in models}


def expected(spec: TaskSpec, converted: list[dict]) -> Values:
    timeout = published(spec).get(SILENCE_TIMEOUT)
    if not timeout or None in timeout.values():
        raise ValueError(f"eot-bench publishes no silence-timeout baseline for {spec.languages[0]}")
    return timeout


def observed(result: dict[str, Any]) -> Values:
    found = result["metrics"].get("cutoffs", {}).get("all", {})
    return {CUTOFF_AT.format(b): (found.get(str(b)) or {}).get("cutoff") for b in LATENCY_BUDGETS_MS}


TURNS = Kind(fits, expected, observed, published)
