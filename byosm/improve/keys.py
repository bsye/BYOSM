from __future__ import annotations

import json
import math
from typing import Any

LOG_BINS_PER_DECADE = 4
EXACT_PATHS = {"train.seed"}


def _bucket(v: Any) -> Any:
    if isinstance(v, str):
        return " ".join(v.lower().split())
    if isinstance(v, bool) or not isinstance(v, int | float) or v == 0:
        return v
    return math.copysign(10 ** (round(math.log10(abs(v)) * LOG_BINS_PER_DECADE) / LOG_BINS_PER_DECADE), v)


def normalized(ops: list[dict]) -> list[dict]:
    out = [{**o, "value": o["value"] if o["path"] in EXACT_PATHS else _bucket(o.get("value"))} for o in ops]
    return sorted(out, key=lambda o: json.dumps(o, sort_keys=True, default=str))


def proposal_key(pr: dict) -> str:
    return json.dumps(normalized(pr["ops"]), sort_keys=True, default=str)
