from __future__ import annotations

import itertools
from collections.abc import Iterator
from typing import Any

from ...rt import OUTSIDE, bio, spans
from ...spec import HFSource, TaskSpec
from ...text import markup_tags
from .decode import _decode
from .hub import _no_decode, open_split
from .parquet import balanced_rows


def class_names(src: HFSource, ds: Any) -> list[str] | None:
    from datasets import ClassLabel

    feat = (ds.features or {}).get(src.label_column) if src.label_column else None
    feat = getattr(feat, "feature", feat)
    return feat.names if isinstance(feat, ClassLabel) else None


def _tag_sequence(raw: list, names: list[str] | None, lmap: dict[str, str], spec: TaskSpec) -> list[str]:
    ours = set(spec.label_names)
    named = [names[t] if names is not None and isinstance(t, int) and t < len(names) else str(t) for t in raw]
    return [lmap.get(t, t if not lmap and t in ours else OUTSIDE) for t in named]


def _row_label(raw: Any, names: list[str] | None, lmap: dict[str, str]) -> str | None:
    keys = [str(raw)] + ([names[raw]] if names is not None and isinstance(raw, int) and raw < len(names) else [])
    return next((lmap[k] for k in keys if k in lmap), None)


def _stream_rows(src: HFSource, spec: TaskSpec) -> Iterator[tuple[Any, str | None]]:
    ds = open_split(src.dataset, src.config, src.split)
    names = class_names(src, ds)
    ds = _no_decode(ds).shuffle(seed=0, buffer_size=1000)
    lmap = {str(a): b for a, b in src.label_map}
    n = 0
    for row in ds:
        lab, raw = None, row[src.input_column]
        if src.format == "markup":
            raw, tags = markup_tags(row[src.label_column])
            lab = _tag_sequence(tags, None, lmap, spec)
        elif src.label_column and isinstance(row[src.label_column], list):
            lab = _tag_sequence(row[src.label_column], names, lmap, spec)
        elif src.label_column:
            lab = _row_label(row[src.label_column], names, lmap)
            if lmap and lab is None:
                continue
        try:
            x = _decode(raw, spec.modality, spec.data.audio.sample_rate)
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


def source_labels(src: HFSource, scan_rows: int) -> list[str]:
    ds = open_split(src.dataset, src.config, src.split)
    if src.format != "markup":
        return class_names(src, ds) or sorted({str(r[src.label_column]) for r in itertools.islice(ds, scan_rows)})
    kinds = {k for r in itertools.islice(ds, scan_rows) for k, _, _ in spans(markup_tags(r[src.label_column])[1])}
    return [OUTSIDE, *(t for k in sorted(kinds) for t in (bio(k, 2)))]
