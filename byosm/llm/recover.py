from __future__ import annotations

import json
import time
from collections.abc import Callable
from typing import Any

from .base import LLM
from .limits import FAIL, LOCAL, is_limit, local_llm_config, settings, wait_out

LIMIT_EVENT = "limit"
ERROR_CHARS = 300
_local: dict[str, LLM] = {}


def local_llm() -> LLM:
    from . import PROVIDERS

    conf = local_llm_config()
    key = json.dumps(conf, sort_keys=True)
    if key not in _local:
        provider = conf.pop("provider")
        _local[key] = PROVIDERS[provider](**conf)
    return _local[key]


def _next_mode(error: Exception, attempt: int, started: float, base: dict[str, Any]) -> str:
    from ..events import emit

    if not is_limit(error):
        return FAIL
    mode = settings()["mode"]
    if mode in (LOCAL, FAIL):
        return mode
    message = str(error)[:ERROR_CHARS]
    return wait_out(error, attempt, started, lambda until: emit(LIMIT_EVENT, **base, until=round(until), error=message))


def through_limits(call: Callable[[LLM], Any], llm: LLM, base: dict[str, Any]) -> Any:
    from ..events import emit

    started, attempt = time.time(), 0
    while True:
        try:
            return call(llm)
        except Exception as e:
            mode = _next_mode(e, attempt, started, base)
            if mode == FAIL or (mode == LOCAL and llm is local_llm()):
                raise
            attempt += 1
            llm = local_llm() if mode == LOCAL else llm
            emit(LIMIT_EVENT, **base, resumed=mode, model_now=getattr(llm, "name", ""))
