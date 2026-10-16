"""Find and map Hugging Face datasets onto a TaskSpec. Audio/images are decoded with soundfile/PIL (no ffmpeg needed)."""
from __future__ import annotations

import io
import json
import itertools
from typing import Any, Callable, Iterator

import numpy as np

from ..llm import get_llm
from ..spec import HFSource, TaskSpec

MAP_SYSTEM = """You map a Hugging Face dataset onto a classifier's label space. Pick the config/split, the input column \
(the raw text/audio/image), and for role=labeled the label column plus an explicit mapping from source label values to \
our labels (omit source values that do not fit any of our labels cleanly - precision beats coverage). For role \
positive/negative/unlabeled only the input column matters (label_map may filter rows: map wanted source values to "keep"). \
Set usable=false if the dataset does not fit the role."""

PICK_SYSTEM = """You choose Hugging Face datasets to build a training set. Given search results, select the datasets \
(by exact id) that best provide data in each role. Prefer widely-used, permissively licensed, parquet-backed datasets."""


def _features(src: HFSource) -> tuple[dict, list[str], dict, list[str]]:
    from datasets import get_dataset_config_names, get_dataset_split_names, load_dataset

    try:
        configs = get_dataset_config_names(src.dataset)
    except Exception:
        configs = []
    cfg = src.config or (configs[0] if configs and configs[0] != "default" else None)
    try:
        splits = get_dataset_split_names(src.dataset, cfg)
    except Exception:
        splits = [src.split]
    ds = load_dataset(src.dataset, cfg, split=src.split if src.split in splits else splits[0], streaming=True)
    feats = ds.features or {}
    row = next(iter(_no_decode(ds)))
    preview = {k: (str(v)[:120] if not isinstance(v, dict) else "<media>") for k, v in row.items()}
    desc = {k: str(v)[:300] for k, v in feats.items()}
    return desc, configs, preview, splits


def _no_decode(ds: Any) -> Any:
    from datasets import Audio, Image

    for k, v in (ds.features or {}).items():
        if isinstance(v, Audio):
            ds = ds.cast_column(k, Audio(decode=False))
        elif isinstance(v, Image):
            ds = ds.cast_column(k, Image(decode=False))
    return ds


def search(queries: list[str], limit: int = 8) -> list[dict]:
    from huggingface_hub import HfApi

    api, seen, out = HfApi(), set(), []
    for q in queries:
        try:
            for d in api.list_datasets(search=q, sort="downloads", limit=limit):
                if d.id not in seen:
                    seen.add(d.id)
                    out.append({"id": d.id, "downloads": getattr(d, "downloads", 0), "tags": [t for t in (d.tags or []) if ":" in t][:8]})
        except Exception:
            continue
    return out


def resolve(spec: TaskSpec, cfg: dict | None = None, log: Callable[[str], None] = print) -> list[HFSource]:
    """Make every source concrete (config/split/columns/label_map), searching the Hub when the spec has none."""
    sources = list(spec.data.hf_sources)
    if not sources and spec.data.hf_queries:
        hits = search(spec.data.hf_queries)
        if hits:
            schema = {"type": "object", "properties": {"picks": {"type": "array", "items": {"type": "object", "properties": {
                "dataset": {"type": "string"}, "role": {"type": "string", "enum": ["labeled", "positive", "negative", "unlabeled"]}}}}}}
            listing = "\n".join(f"- {h['id']} (downloads={h['downloads']}, {', '.join(h['tags'])})" for h in hits)
            out = get_llm("spec", cfg).json(PICK_SYSTEM, f"Task: {spec.summary or spec.description}\nModality: {spec.modality}\nLabels: {spec.label_names}\n\nSearch results:\n{listing}", schema)
            ids = {h["id"] for h in hits}
            sources = [HFSource(dataset=p["dataset"], role=p["role"]) for p in out.get("picks", []) if p["dataset"] in ids]
    mapped = []
    schema = {"type": "object", "properties": {
        "usable": {"type": "boolean"}, "config": {"type": "string"}, "split": {"type": "string"},
        "input_column": {"type": "string"}, "label_column": {"type": "string"},
        "label_map": {"type": "array", "items": {"type": "array", "items": {"type": "string"}}}}}
    for src in sources:
        if src.input_column:
            mapped.append(src)
            continue
        try:
            feats, configs, preview, splits = _features(src)
        except Exception as e:
            log(f"  skip {src.dataset}: {type(e).__name__}: {str(e)[:120]}")
            continue
        prompt = (f"Our task: {spec.summary or spec.description}\nOur labels: {[(l.name, l.definition) for l in spec.labels]}\n"
                  f"Role of this dataset: {src.role}\nDataset: {src.dataset}\nConfigs: {configs[:30]}\nSplits: {splits}\n"
                  f"Features: {feats}\nFirst row: {preview}")
        out = get_llm("spec", cfg).json(MAP_SYSTEM, prompt, schema)
        if not out.get("usable", True) or out.get("input_column") not in feats:
            log(f"  skip {src.dataset}: not usable for role {src.role}")
            continue
        src = src.model_copy(update={
            "config": out.get("config") or src.config if (out.get("config") in configs) else src.config,
            "split": out.get("split") if out.get("split") in splits else splits[0],
            "input_column": out["input_column"],
            "label_column": out.get("label_column") or None if out.get("label_column") in feats else None,
            "label_map": [m for m in out.get("label_map", []) if len(m) == 2],
        })
        log(f"  mapped {src.dataset} [{src.config or '-'}:{src.split}] role={src.role} input={src.input_column} label={src.label_column} map={len(src.label_map)}")
        mapped.append(src)
    return mapped


def _decode(v: Any, modality: str, sr: int) -> Any:
    if modality == "text":
        return str(v)
    if modality == "audio":
        import soundfile as sf

        if isinstance(v, dict) and v.get("bytes"):
            wav, file_sr = sf.read(io.BytesIO(v["bytes"]), dtype="float32", always_2d=True)
        elif isinstance(v, dict) and "array" in v:
            wav, file_sr = np.asarray(v["array"], dtype=np.float32)[:, None], v["sampling_rate"]
        else:
            wav, file_sr = sf.read(v["path"], dtype="float32", always_2d=True)
        wav = wav.mean(1)
        if file_sr != sr:
            import torch
            import torchaudio

            wav = torchaudio.functional.resample(torch.from_numpy(wav), file_sr, sr).numpy()
        return wav.astype(np.float32)
    if modality in ("image", "document"):
        from PIL import Image

        if isinstance(v, dict) and v.get("bytes"):
            return Image.open(io.BytesIO(v["bytes"])).convert("RGB")
        return Image.open(v["path"]).convert("RGB")
    raise ValueError(modality)


def _norm(s: str) -> str:
    return " ".join(str(s).lower().replace("_", " ").replace("-", " ").split())


def balanced_rows(src: HFSource, spec: TaskSpec, seed: int = 0) -> list[tuple[Any, str]]:
    """`per_label` random rows per label from the split's parquet files (downloaded once, cached), reading only the
    label column to index and only the chosen rows' inputs. Works on label-sorted datasets where streaming can't."""
    import random

    import pyarrow.parquet as pq
    from huggingface_hub import HfApi, hf_hub_download

    files = sorted(f for f in HfApi().list_repo_files(src.dataset, repo_type="dataset")
                   if f.endswith(".parquet") and (f.split("/")[-1].startswith(f"{src.split}-") or f"/{src.split}/" in f))
    if not files:
        raise ValueError(f"{src.dataset}: no parquet files for split {src.split!r}")
    names: list[str] | None = None
    lmap = {_norm(a): b for a, b in src.label_map}
    ours = {_norm(n): n for n in spec.label_names}
    by_label: dict[str, list[tuple[str, int]]] = {}
    for f in files:
        path = hf_hub_download(src.dataset, f, repo_type="dataset")
        pf = pq.ParquetFile(path)
        if names is None:  # ClassLabel names: parquet metadata if present, else the dataset card
            meta = json.loads((pf.schema_arrow.metadata or {}).get(b"huggingface", b"{}"))
            names = meta.get("info", {}).get("features", {}).get(src.label_column, {}).get("names") or []
            if not names:
                from datasets import ClassLabel, load_dataset_builder

                feat = (load_dataset_builder(src.dataset, src.config).info.features or {}).get(src.label_column)
                names = feat.names if isinstance(feat, ClassLabel) else []
        for i, v in enumerate(pf.read(columns=[src.label_column]).column(src.label_column).to_pylist()):
            raw = names[v] if names and isinstance(v, int) and v < len(names) else str(v)
            lab = lmap.get(_norm(raw)) or lmap.get(str(v)) or (ours.get(_norm(raw)) if not lmap else None)
            if lab is None:
                continue
            if src.partition != "all":
                from .segments import partition_of

                if partition_of(f"{f}:{i}") != src.partition:
                    continue
            by_label.setdefault(lab, []).append((path, i))
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


def iter_rows(src: HFSource, spec: TaskSpec) -> Iterator[tuple[Any, str | None]]:
    """Yield (decoded_input, our_label_or_None). Rows whose label is not in label_map are skipped."""
    from datasets import ClassLabel, load_dataset

    if src.sampling == "balanced":
        yield from balanced_rows(src, spec)
        return

    ds = load_dataset(src.dataset, src.config, split=src.split, streaming=True)
    names = None
    if src.label_column and isinstance((ds.features or {}).get(src.label_column), ClassLabel):
        names = ds.features[src.label_column].names
    ds = _no_decode(ds).shuffle(seed=0, buffer_size=1000)
    lmap = {str(a): b for a, b in src.label_map}
    n = 0
    for row in ds:
        lab = None
        if src.label_column:
            raw = row[src.label_column]
            keys = [str(raw)] + ([names[raw]] if names is not None and isinstance(raw, int) and raw < len(names) else [])
            lab = next((lmap[k] for k in keys if k in lmap), None)
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


def take(src: HFSource, spec: TaskSpec, n: int | None = None) -> list[tuple[Any, str | None]]:
    if src.sampling == "balanced":
        return balanced_rows(src, spec)
    return list(itertools.islice(iter_rows(src, spec), n or src.max_rows))
