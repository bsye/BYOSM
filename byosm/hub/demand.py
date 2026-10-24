from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

REPORT = "hub:demand"
TOP = 50
MIN_DEMAND = 100_000
TASKS = {
    "text-classification": "text classification",
    "token-classification": "token tagging",
    "audio-classification": "audio classification",
    "image-classification": "image classification",
    "time-series-forecasting": "time series forecasting",
}
LANGUAGE_FREE = {"image-classification", "time-series-forecasting"}
LANGUAGES = {
    "en": "English",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "it": "Italian",
    "pt": "Portuguese",
    "nl": "Dutch",
    "pl": "Polish",
    "ru": "Russian",
    "ar": "Arabic",
    "zh": "Chinese",
    "ja": "Japanese",
    "ko": "Korean",
    "hi": "Hindi",
    "tr": "Turkish",
}


def _slot(api: Any, task: str, language: str | None) -> dict[str, Any]:
    models = list(api.list_models(pipeline_tag=task, filter=language, sort="downloads", limit=TOP))
    datasets = list(api.list_datasets(task_categories=task, language=language, sort="downloads", limit=TOP))
    demand = sum(m.downloads or 0 for m in models)
    supply = sum(d.downloads or 0 for d in datasets)
    return {
        "task": task,
        "language": language,
        "demand": demand,
        "supply": supply,
        "datasets": len(datasets),
        "score": round(demand / (supply + 1), 2),
        "top_model": models[0].id if models else None,
        "top_dataset": datasets[0].id if datasets else None,
        "prompt": f"{TASKS[task]}" + (f" in {LANGUAGES[language]}" if language else ""),
    }


def scan(log: Callable[[str], None]) -> list[dict[str, Any]]:
    from huggingface_hub import HfApi

    api, rows = HfApi(), []
    for task in TASKS:
        for language in [None] if task in LANGUAGE_FREE else list(LANGUAGES):
            row = _slot(api, task, language)
            log(
                f"  {row['prompt']}: models {row['demand']:,} downloads, datasets {row['supply']:,} ({row['datasets']})"
            )
            rows.append(row)
    return sorted((r for r in rows if r["demand"] >= MIN_DEMAND), key=lambda r: -r["score"])


def save(ws: Path, rows: list[dict[str, Any]]) -> None:
    from ..db import get_or_add, session
    from ..db.models import WorkspaceReport

    with session(ws) as s:
        get_or_add(s, WorkspaceReport, kind=REPORT).data = {"at": time.time(), "rows": rows}


def latest(ws: Path) -> dict[str, Any]:
    from sqlalchemy import select

    from ..db import session
    from ..db.models import WorkspaceReport

    with session(ws) as s:
        return s.scalar(select(WorkspaceReport.data).where(WorkspaceReport.kind == REPORT)) or {"at": None, "rows": []}
