from __future__ import annotations

from numbers import Real
from typing import Any

MISSING = "–"


def fmt(x: Any, nd: int = 4, missing: str = MISSING) -> str:
    if x is None:
        return missing
    return f"{x:.{nd}f}" if isinstance(x, Real) and not isinstance(x, bool) else str(x)


def params(n: int | None, missing: str = "-") -> str:
    return f"{n / 1e6:.2f}M" if n else missing


def cpu(r: dict) -> str:
    if r.get("stream_ms"):
        return f"{r['stream_ms']:.3f}/{fmt(r.get('chunk_ms'), 0, '?')}ms"
    return f"{r['latency_ms']:.2f}ms" if r.get("latency_ms") else "-"


def interval(ci: dict | None, nd: int = 3, sep: str = "-") -> str:
    return f"{ci['lo']:.{nd}f}{sep}{ci['hi']:.{nd}f}" if ci else ""
