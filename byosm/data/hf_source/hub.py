from __future__ import annotations

from typing import Any

from ...spec import HFSource

PARQUET_BRANCH = "refs/convert/parquet"
SCRIPT_ERROR = "Dataset scripts are no longer supported"


def hub_call(fn: Any, dataset: str, config: str | None, **kw: Any) -> Any:
    try:
        return fn(dataset, config, **kw)
    except RuntimeError as e:
        if SCRIPT_ERROR not in str(e):
            raise
        return fn(dataset, data_dir=config, revision=PARQUET_BRANCH, **kw)


def open_split(dataset: str, config: str | None, split: str) -> Any:
    from datasets import load_dataset

    return hub_call(load_dataset, dataset, config, split=split, streaming=True)


def _no_decode(ds: Any) -> Any:
    from datasets import Audio, Image

    for k, v in (ds.features or {}).items():
        if isinstance(v, Audio):
            ds = ds.cast_column(k, Audio(decode=False))
        elif isinstance(v, Image):
            ds = ds.cast_column(k, Image(decode=False))
    return ds


def _features(src: HFSource) -> tuple[dict, list[str], dict, list[str]]:
    from datasets import get_dataset_config_names, get_dataset_split_names

    try:
        configs = get_dataset_config_names(src.dataset)
    except Exception:
        configs = []
    cfg = src.config or (configs[0] if configs and configs[0] != "default" else None)
    try:
        splits = hub_call(get_dataset_split_names, src.dataset, cfg)
    except Exception:
        splits = [src.split]
    ds = open_split(src.dataset, cfg, src.split if src.split in splits else splits[0])
    feats = ds.features or {}
    row = next(iter(_no_decode(ds)))
    preview = {k: (str(v)[:120] if not isinstance(v, dict) else "<media>") for k, v in row.items()}
    desc = {k: str(v)[:300] for k, v in feats.items()}
    return desc, configs, preview, splits


def _hit(d: Any) -> dict:
    return {"id": d.id, "downloads": getattr(d, "downloads", 0), "tags": [t for t in (d.tags or []) if ":" in t][:8]}


SEARCH_FILLER = {
    "dataset",
    "datasets",
    "benchmark",
    "benchmarks",
    "model",
    "models",
    "huggingface",
    "hf",
    "classification",
}


def search_terms(queries: list[str]) -> list[str]:
    terms = []
    for q in queries:
        words = [w for w in q.split() if w.lower() not in SEARCH_FILLER]
        for t in [q, " ".join(words), *words]:
            if t and t not in terms:
                terms.append(t)
    return terms


def search(queries: list[str], limit: int = 8) -> list[dict]:
    from huggingface_hub import HfApi

    api, seen, out = HfApi(), set(), []
    for q in search_terms(queries):
        try:
            for d in api.list_datasets(search=q, sort="downloads", limit=limit):
                if d.id not in seen:
                    seen.add(d.id)
                    out.append(_hit(d))
        except Exception:
            continue
    return out


def exists(dataset: str) -> bool:
    from huggingface_hub import HfApi
    from huggingface_hub.errors import RepositoryNotFoundError

    try:
        HfApi().dataset_info(dataset)
    except RepositoryNotFoundError:
        return False
    except Exception:
        return True
    return True


def split_sizes(dataset: str, config: str | None = None) -> dict[str, int]:
    try:
        from datasets import load_dataset_builder

        splits = hub_call(load_dataset_builder, dataset, config).info.splits or {}
        return {k: int(v.num_examples) for k, v in splits.items() if v.num_examples}
    except Exception:
        return {}
