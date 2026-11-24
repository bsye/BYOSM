from __future__ import annotations

import os
import shutil
import tarfile
from pathlib import Path

from ....split_file import GB, Log, checked, held, joined, reporting, sums
from . import BUCKETS, DATA, WEIGHTS, Blueprint, blueprint, work_dir

PART_SUMS = "SHA256SUMS.parts"
FILE_SUMS = "SHA256SUMS.models"
ARCHIVE = f"{BUCKETS}.tar"
PARTS = "parts"
UNPACKING = ".unpacking"
RESERVE_GB = 40


def download(found: Blueprint, folder: Path, name: str) -> Path:
    from huggingface_hub import constants, hf_hub_download

    constants.HF_HUB_DISABLE_XET = True
    return Path(hf_hub_download(found.repo, name, revision=found.revision, local_dir=folder / PARTS, token=False))


def sizes(found: Blueprint) -> dict[str, int]:
    from huggingface_hub import HfApi

    info = HfApi(token=False).model_info(found.repo, revision=found.revision, files_metadata=True)
    return {s.rfilename: s.size for s in info.siblings}


def room(folder: Path, needed: float) -> None:
    free = shutil.disk_usage(folder).free
    if free < needed + RESERVE_GB * GB:
        raise RuntimeError(
            f"{free / GB:.0f} GB free on the disk holding {folder}: the blueprint still needs {needed / GB:.0f} GB "
            f"and {RESERVE_GB} GB must stay free"
        )


def published(found: Blueprint, folder: Path) -> dict[str, str]:
    return {
        **sums(download(found, folder, PART_SUMS).read_text()),
        **sums(download(found, folder, FILE_SUMS).read_text()),
    }


def buckets(found: Blueprint, root: Path, known: dict[str, str]) -> Path:
    out = root / DATA / BUCKETS
    if out.exists():
        return out
    archive = checked(download(found, root / WEIGHTS, ARCHIVE), known[ARCHIVE])
    unpacking = root / f"{DATA}{UNPACKING}"
    shutil.rmtree(unpacking, ignore_errors=True)
    with tarfile.open(archive) as tar:
        tar.extractall(unpacking, filter="data")
    out.parent.mkdir(parents=True, exist_ok=True)
    os.replace(unpacking / BUCKETS, out)
    shutil.rmtree(unpacking)
    archive.unlink()
    return out


def fetch(ws: Path, name: str, log: Log) -> dict:
    from huggingface_hub.utils import disable_progress_bars

    disable_progress_bars()
    found, root = blueprint(name), work_dir(ws)
    folder = root / WEIGHTS
    folder.mkdir(exist_ok=True)
    known, final = published(found, folder), folder / found.file
    if not final.exists():
        listed, names = sizes(found), found.part_names()
        total = sum(listed[n] for n in names)
        room(folder, total + max(listed[n] for n in names) - held(folder))
        log(f"fetching {name}: {total / GB:.1f} GB in {len(names)} parts from {found.repo}")
        with reporting(folder, total, log):
            joined(folder, found.file, names, known, lambda part: download(found, folder, part), log)
    cards = buckets(found, root, known)
    log(f"{name} is ready: {final.stat().st_size / GB:.1f} GB at {final}, card buckets at {cards}")
    return {"blueprint": name, "file": str(final), "buckets": str(cards), "gb": round(final.stat().st_size / GB, 1)}
