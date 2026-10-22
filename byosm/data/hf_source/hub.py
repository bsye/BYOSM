from __future__ import annotations

from typing import Any

from ...spec import HFSource

PARQUET_BRANCH = "refs/convert/parquet"
SCRIPT_ERROR = "Dataset scripts are no longer supported"


CLOSEST_HITS = 10
PHRASE_WEIGHT = 3
MIN_NAME_LETTERS = 5
NAME_COVER = 0.6


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
    "classify",
    "classifier",
    "detection",
    "detect",
    "prediction",
    "predict",
    "task",
    "label",
    "labels",
    "labelled",
    "english",
    "text",
    "data",
    "corpus",
    "from",
    "with",
    "that",
    "this",
    "into",
    "whether",
    "user",
    "users",
}
MIN_KEYWORD_LETTERS = 4


def keywords(queries: list[str]) -> list[str]:
    words = (w.strip(".,:;()'\"").lower() for q in queries for w in q.replace("-", " ").replace("_", " ").split())
    return list(dict.fromkeys(w for w in words if len(w) >= MIN_KEYWORD_LETTERS and w not in SEARCH_FILLER))


def search_terms(queries: list[str]) -> list[str]:
    phrases = [" ".join(w for w in q.split() if w.lower() not in SEARCH_FILLER) for q in queries]
    return list(dict.fromkeys([*(p for p in phrases if p), *keywords(queries)]))


def relevance(dataset: str, queries: list[str]) -> int:
    name = _letters(dataset)
    phrases = sum(_letters(q) in name for q in queries if _letters(q))
    return sum(w in name for w in keywords(queries)) + PHRASE_WEIGHT * phrases


def search(queries: list[str], limit: int = 8) -> list[dict]:
    from huggingface_hub import HfApi

    api, found = HfApi(), {}
    for q in search_terms(queries):
        try:
            for d in api.list_datasets(search=q, sort="downloads", limit=limit):
                found.setdefault(d.id, _hit(d))
        except Exception:
            continue
    ranked = sorted(found.values(), key=lambda h: (-relevance(h["id"], queries), -(h["downloads"] or 0)))
    return [h for h in ranked if relevance(h["id"], queries) > 0]


def exists(dataset: str) -> bool:
    from huggingface_hub import HfApi
    from huggingface_hub.errors import HFValidationError, RepositoryNotFoundError

    try:
        HfApi().dataset_info(dataset)
    except (RepositoryNotFoundError, HFValidationError):
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


def _letters(s: str) -> str:
    return "".join(c for c in s.lower() if c.isalnum())


def closest(dataset: str) -> str | None:
    from huggingface_hub import HfApi

    if exists(dataset):
        return dataset
    name = dataset.split("/")[-1]
    key = _letters(name)
    try:
        hits = list(HfApi().list_datasets(search=name, sort="downloads", limit=CLOSEST_HITS))
    except Exception:
        return None

    def same(hit: str) -> bool:
        theirs = _letters(hit.split("/")[-1])
        return len(key) >= MIN_NAME_LETTERS and key in theirs and len(key) >= NAME_COVER * len(theirs)

    return next((h.id for h in hits if same(h.id)), None)
