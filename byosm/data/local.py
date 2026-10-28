from __future__ import annotations

import random
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ..spec import TaskSpec
from .splits import HELD, _split3
from .store import TEST_SHARE, record
from .table import is_table, read_table, table_rows

Log = Callable[[str], None]


def local_rows(spec: TaskSpec) -> list[tuple[Any, Any, str]]:
    root = Path(spec.data.local_dir or "")
    if is_table(root):
        return table_rows(spec, root)
    out = []
    for lab in spec.label_names:
        for p in sorted((root / lab).glob("*")) if (root / lab).exists() else []:
            if p.suffix == ".txt" and spec.modality == "text":
                out += [(line.strip(), lab, "") for line in p.read_text().splitlines() if line.strip()]
            elif spec.modality != "text":
                out.append((str(p.resolve()), lab, ""))
    return out


def add_local(spec: TaskSpec, splits: dict[str, list[dict]], rng: random.Random, log: Log) -> None:
    local = local_rows(spec)
    source = spec.data.local_source
    marked = {m: [record(x, y, source) for x, y, s in local if s == m] for m in HELD}
    rows = [record(x, y, source) for x, y, s in local if s not in HELD]
    shares = {m: 0.0 for m in HELD if marked[m]}
    parts = _split3(rows, rng, **shares) if shares else _split3(rows, rng, test=TEST_SHARE)
    for s, part in parts.items():
        splits[s] += part + marked.get(s, [])
    counts = ", ".join(f"{len(v)} marked {m}" for m, v in marked.items() if v)
    log(f"  local: {len(local)} examples from {spec.data.local_dir} ({counts or 'none marked'})")


def add_pairs(spec: TaskSpec, splits: dict[str, list[dict]], cfg: dict | None, log: Log) -> None:
    from ..options.training import pair_splits

    for s, rows in pair_splits(spec, read_table(Path(spec.data.local_dir or "")), cfg, log).items():
        splits[s] += rows
