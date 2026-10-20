from __future__ import annotations

import json
from typing import Any


def _numeric(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) or isinstance(x, list) and all(map(_numeric, x))


def _dumps(o: Any, ind: int = 0) -> str:
    pad = "  " * (ind + 1)
    end = "\n" + "  " * ind
    if isinstance(o, dict) and o:
        return "{\n" + ",\n".join(f"{pad}{json.dumps(str(k))}: {_dumps(v, ind + 1)}" for k, v in o.items()) + end + "}"
    if isinstance(o, list) and o and not _numeric(o):
        return "[\n" + ",\n".join(pad + _dumps(v, ind + 1) for v in o) + end + "]"
    return json.dumps(o, separators=(",", ":")) if isinstance(o, list) else json.dumps(o)
