from __future__ import annotations

from pathlib import Path

from ..data.store import CALIB
from ..options.heldout import hashed

CALIB_SHARE = 0.2
TEST = "test"


def split_of(key: str) -> str:
    return CALIB if hashed(key) < CALIB_SHARE else TEST


def download(repo: str, file: str) -> Path:
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(repo, file, repo_type="dataset"))
