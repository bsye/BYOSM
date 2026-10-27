from __future__ import annotations

import json
import random
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path

REPO = "Salesforce/GiftEvalPretrain"
ARROW = ".arrow"
SEED = 0
MANIFEST = "corpus.json"
GB = 1e9
CORPUS_GB = 50.0
PRETRAIN_STEPS = 50_000


def _manifest() -> Path:
    from ..config import cache_root

    return cache_root() / "series" / MANIFEST


def _listing() -> dict[str, list[tuple[str, int]]]:
    from huggingface_hub import HfApi

    by: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for s in HfApi().dataset_info(REPO, files_metadata=True).siblings:
        if s.rfilename.endswith(ARROW):
            by[s.rfilename.split("/")[0]].append((s.rfilename, s.size or 0))
    return by


def plan(listing: dict[str, list[tuple[str, int]]], budget: float) -> list[str]:
    rng, chosen, left = random.Random(SEED), [], budget
    ordered = sorted(listing, key=lambda d: sum(size for _, size in listing[d]))
    for i, name in enumerate(ordered):
        share, taken = left / (len(ordered) - i), 0
        files = listing[name][:]
        rng.shuffle(files)
        for path, size in files:
            if taken + size <= share or not taken and size <= 2 * share:
                chosen.append(path)
                taken += size
        left -= taken
    return chosen


def download(gb: float, log: Callable[[str], None]) -> list[str]:
    from huggingface_hub import hf_hub_download

    listing = _listing()
    sizes = {p: s for files in listing.values() for p, s in files}
    chosen = plan(listing, gb * GB)
    log(
        f"  {len(chosen)} files from {len({p.split('/')[0] for p in chosen})} datasets, {sum(sizes[p] for p in chosen) / GB:.1f} GB"
    )
    paths, done = [], 0.0
    for i, path in enumerate(chosen):
        paths.append(hf_hub_download(REPO, path, repo_type="dataset"))
        done += sizes[path]
        log(f"  {i + 1}/{len(chosen)} {path}: {done / GB:.1f} GB so far")
    _manifest().parent.mkdir(parents=True, exist_ok=True)
    _manifest().write_text(json.dumps(paths))
    return paths


def files() -> list[str]:
    return json.loads(_manifest().read_text()) if _manifest().exists() else []
