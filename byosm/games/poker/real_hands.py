from __future__ import annotations

import ast
import hashlib
import shutil
import urllib.request
import zipfile
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

from ...config import STATE_DIR
from ...options.heldout import hashed

RECORD = "13997158"
ARCHIVE = "poker-hand-histories.zip"
MD5 = "0918a30f97bda0129897b7e0f1ec895a"
URL = f"https://zenodo.org/api/records/{RECORD}/files/{ARCHIVE}/content"
LICENCE = "cc-by-4.0"
FOLDER = Path("datasets") / f"zenodo-{RECORD}"
PREFIX = "data/handhq/"
SETS = ".phhs"
STAKE_SUFFIX = "NLH"
SAMPLE_KEY = "real-hands:"
WANTED = ("variant", "antes", "blinds_or_straddles", "starting_stacks", "actions", "players")
AMOUNTS = ("antes", "blinds_or_straddles", "starting_stacks")
NOT_LOGGED, AS_NUMBER = "inf", "1e999"
SITES = {
    "ABS": "Absolute Poker",
    "FTP": "Full Tilt Poker",
    "IPN": "iPoker",
    "ONG": "Ongame",
    "PS": "PokerStars",
    "PTY": "PartyPoker",
}
BUFFER = 1 << 22
Log = Callable[[str], None]


def archive(ws: Path) -> Path:
    return ws / STATE_DIR / FOLDER / ARCHIVE


def md5(path: Path) -> str:
    with path.open("rb") as f:
        return hashlib.file_digest(f, "md5").hexdigest()


def fetched(ws: Path, log: Log) -> Path:
    path = archive(ws)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        log(f"downloading {ARCHIVE} from Zenodo record {RECORD} ({LICENCE})")
        with urllib.request.urlopen(URL) as source, path.with_suffix(".partial").open("wb") as out:
            shutil.copyfileobj(source, out, BUFFER)
        path.with_suffix(".partial").rename(path)
    found = md5(path)
    if found != MD5:
        raise ValueError(f"{path} has md5 {found}, not the published {MD5}: delete it and fetch again")
    return path


def named(entry: str) -> tuple[str, str]:
    folder = entry.removeprefix(PREFIX).partition("/")[0]
    stake = next(part for part in folder.split("_") if part.endswith(STAKE_SUFFIX))
    return folder.partition("-")[0], stake.removesuffix(STAKE_SUFFIX)


def sections(text: str) -> Iterator[dict[str, Any]]:
    hand: dict[str, Any] = {}
    for line in text.splitlines():
        if line.startswith("[") and line.endswith("]"):
            if hand:
                yield hand
            hand = {}
            continue
        key, sep, value = line.partition(" = ")
        if sep and key in WANTED:
            hand[key] = ast.literal_eval(value.replace(NOT_LOGGED, AS_NUMBER) if key in AMOUNTS else value)
    if hand:
        yield hand


def sampled(names: list[str], share: float) -> list[str]:
    return sorted(n for n in names if n.startswith(PREFIX) and n.endswith(SETS) and hashed(SAMPLE_KEY + n) < share)


def hands(ws: Path, share: float) -> Iterator[tuple[str, dict[str, Any]]]:
    path = archive(ws)
    if not path.exists():
        raise ValueError(f"{path} is not here yet: run byosm poker real-hands-fetch first")
    with zipfile.ZipFile(path) as held:
        for entry in sampled(held.namelist(), share):
            site, stake = named(entry)
            text = held.read(entry).decode("utf-8", "replace")
            for number, hand in enumerate(sections(text), start=1):
                yield f"{site}-{stake}/{Path(entry).stem}#{number}", hand
