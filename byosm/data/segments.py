from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from ..spec import HFSource, TaskSpec


def partition_of(key: str) -> str:
    h = int(hashlib.md5(key.encode()).hexdigest(), 16) % 100
    return "train" if h < 70 else ("calib" if h < 76 else "test")


def list_files(src: HFSource) -> list[str]:
    from huggingface_hub import HfApi

    files = [
        f.path
        for f in HfApi().list_repo_tree(src.dataset, repo_type="dataset", recursive=True)
        if getattr(f, "path", "").endswith((".wav", ".flac"))
    ]
    return sorted(files)


def fetch(src: HFSource, partitions: tuple[str, ...], max_files: int, cache: Path) -> list[Path]:
    from huggingface_hub import snapshot_download

    files = [f for f in list_files(src) if partition_of(Path(f).name) in partitions][:max_files]
    pats = [p for f in files for p in (f, str(Path(f).with_suffix(".json")))]
    root = Path(
        snapshot_download(
            src.dataset, repo_type="dataset", allow_patterns=pats, local_dir=str(cache / src.dataset.replace("/", "__"))
        )
    )
    return [root / f for f in files]


def frame_labels(wav_path: Path, n_frames: int, spec: TaskSpec, label_filter: list[str] | None = None) -> np.ndarray:
    meta = json.loads(wav_path.with_suffix(".json").read_text())
    hop_s = spec.frame_hop_ms / 1000
    y = np.zeros(n_frames, dtype=np.int64)
    clusters = meta.get("cluster") or [None] * len(meta["onset"])
    for on, off, c in zip(meta["onset"], meta["offset"], clusters, strict=False):
        if label_filter and c not in label_filter:
            continue
        y[int(round(on / hop_s)) : int(round(off / hop_s))] = 1
    neg = next(i for i in range(len(spec.labels)) if i != spec.pos_index)
    return np.where(y > 0, spec.pos_index, neg)
