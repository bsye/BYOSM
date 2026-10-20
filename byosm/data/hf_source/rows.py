from __future__ import annotations

import itertools
from collections.abc import Iterator
from typing import Any

from ...spec import HFSource, TaskSpec
from .decode import _decode
from .hub import _no_decode
from .parquet import balanced_rows


def _class_names(src: HFSource, ds: Any) -> list[str] | None:
    from datasets import ClassLabel

    if src.label_column and isinstance((ds.features or {}).get(src.label_column), ClassLabel):
        return ds.features[src.label_column].names
    return None


def _row_label(raw: Any, names: list[str] | None, lmap: dict[str, str]) -> str | None:
    keys = [str(raw)] + ([names[raw]] if names is not None and isinstance(raw, int) and raw < len(names) else [])
    return next((lmap[k] for k in keys if k in lmap), None)


def _stream_rows(src: HFSource, spec: TaskSpec) -> Iterator[tuple[Any, str | None]]:
    from datasets import load_dataset

    ds = load_dataset(src.dataset, src.config, split=src.split, streaming=True)
    names = _class_names(src, ds)
    ds = _no_decode(ds).shuffle(seed=0, buffer_size=1000)
    lmap = {str(a): b for a, b in src.label_map}
    n = 0
    for row in ds:
        lab = None
        if src.label_column:
            lab = _row_label(row[src.label_column], names, lmap)
            if lmap and lab is None:
                continue
        try:
            x = _decode(row[src.input_column], spec.modality, spec.data.audio.sample_rate)
        except Exception:
            continue
        yield x, lab
        n += 1
        if n >= src.max_rows:
            break


def iter_rows(src: HFSource, spec: TaskSpec) -> Iterator[tuple[Any, str | None]]:
    if src.sampling == "balanced":
        yield from balanced_rows(src, spec)
        return
    yield from _stream_rows(src, spec)


def take(src: HFSource, spec: TaskSpec, n: int | None = None) -> list[tuple[Any, str | None]]:
    if src.sampling == "balanced":
        return balanced_rows(src, spec)
    return list(itertools.islice(iter_rows(src, spec), n or src.max_rows))
