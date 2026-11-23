from __future__ import annotations

import subprocess
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


def sparse_clone(url: str, commit: str, folder: str, dest: Path) -> Path:
    found = dest / folder
    if found.exists():
        return found
    dest.parent.mkdir(parents=True, exist_ok=True)
    git = ["git", "-C", str(dest)]
    subprocess.run(
        ["git", "clone", "-q", "--filter=blob:none", "--sparse", "--no-checkout", url, str(dest)], check=True
    )
    subprocess.run([*git, "sparse-checkout", "set", folder], check=True)
    subprocess.run([*git, "checkout", "-q", commit], check=True)
    return found
