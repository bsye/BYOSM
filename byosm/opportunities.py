from __future__ import annotations

import time
from functools import lru_cache
from typing import Any

TASKS = ("text-classification", "token-classification", "audio-classification", "image-classification")
PER_TASK = 4
REFRESH_S = 3600
FIELDS = ["downloads", "likes", "pipeline_tag", "safetensors", "trendingScore"]


def _row(m: Any) -> dict[str, Any]:
    return {
        "id": m.id,
        "task": m.pipeline_tag,
        "downloads": m.downloads,
        "likes": m.likes,
        "trending": getattr(m, "trending_score", None) or 0,
        "params": m.safetensors.total if m.safetensors else None,
    }


@lru_cache(maxsize=1)
def _cached(_bucket: int) -> list[dict[str, Any]]:
    from huggingface_hub import HfApi

    api = HfApi()
    found = [
        _row(m)
        for task in TASKS
        for m in api.list_models(pipeline_tag=task, sort="trending_score", limit=PER_TASK, expand=FIELDS)
    ]
    return sorted(found, key=lambda r: -r["trending"])


def trending() -> list[dict[str, Any]]:
    return _cached(int(time.time() // REFRESH_S))
