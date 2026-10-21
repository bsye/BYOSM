from __future__ import annotations

import json
import random
from typing import Any

from ...spec import HFSource, TaskSpec
from ...text import normalize
from .decode import _decode
from .hub import hub_call


def _split_files(src: HFSource) -> list[str]:
    from huggingface_hub import HfApi

    files = sorted(
        f
        for f in HfApi().list_repo_files(src.dataset, repo_type="dataset")
        if f.endswith(".parquet") and (f.split("/")[-1].startswith(f"{src.split}-") or f"/{src.split}/" in f)
    )
    if not files:
        raise ValueError(f"{src.dataset}: no parquet files for split {src.split!r}")
    return files


def _class_names(src: HFSource, pf: Any) -> list[str]:
    meta = json.loads((pf.schema_arrow.metadata or {}).get(b"huggingface", b"{}"))
    names = meta.get("info", {}).get("features", {}).get(src.label_column, {}).get("names") or []
    if names:
        return names
    from datasets import ClassLabel, load_dataset_builder

    feat = (hub_call(load_dataset_builder, src.dataset, src.config).info.features or {}).get(src.label_column)
    return feat.names if isinstance(feat, ClassLabel) else []


def _in_partition(src: HFSource, key: str) -> bool:
    if src.partition == "all":
        return True
    from ..segments import partition_of

    return partition_of(key) == src.partition


def _label_index(src: HFSource, spec: TaskSpec) -> dict[str, list[tuple[str, int]]]:
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    files = _split_files(src)
    names: list[str] | None = None
    lmap = {normalize(a): b for a, b in src.label_map}
    ours = {normalize(n): n for n in spec.label_names}
    by_label: dict[str, list[tuple[str, int]]] = {}
    for f in files:
        path = hf_hub_download(src.dataset, f, repo_type="dataset")
        pf = pq.ParquetFile(path)
        if names is None:
            names = _class_names(src, pf)
        for i, v in enumerate(pf.read(columns=[src.label_column]).column(src.label_column).to_pylist()):
            raw = names[v] if names and isinstance(v, int) and v < len(names) else str(v)
            lab = lmap.get(normalize(raw)) or lmap.get(str(v)) or (ours.get(normalize(raw)) if not lmap else None)
            if lab is None or not _in_partition(src, f"{f}:{i}"):
                continue
            by_label.setdefault(lab, []).append((path, i))
    return by_label


def availability(src: HFSource, spec: TaskSpec) -> dict[str, int]:
    return {k: len(v) for k, v in _label_index(src, spec).items()}


def balanced_rows(src: HFSource, spec: TaskSpec, seed: int = 0) -> list[tuple[Any, str]]:
    import pyarrow.parquet as pq

    by_label = _label_index(src, spec)
    rng = random.Random(seed)
    chosen: dict[str, list[tuple[int, str]]] = {}
    for lab, items in by_label.items():
        for path, i in rng.sample(items, min(src.per_label, len(items))):
            chosen.setdefault(path, []).append((i, lab))
    out = []
    for path, rows in chosen.items():
        col = pq.read_table(path, columns=[src.input_column]).column(src.input_column)
        for i, lab in rows:
            try:
                out.append((_decode(col[i].as_py(), spec.modality, spec.data.audio.sample_rate), lab))
            except Exception:
                continue
    rng.shuffle(out)
    return out
