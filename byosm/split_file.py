from __future__ import annotations

import contextlib
import hashlib
import os
import shutil
import threading
from collections.abc import Callable, Iterator
from pathlib import Path

BUFFER = 1 << 24
PARTIAL = ".partial"
JOINED = ".joined"
REPORT_SECONDS = 300
GB = 1e9
Log = Callable[[str], None]


def sums(text: str) -> dict[str, str]:
    found = {}
    for line in text.splitlines():
        digest, _, name = line.strip().partition(" ")
        if name:
            found[name.strip()] = digest
    return found


def sha256(path: Path) -> str:
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def checked(path: Path, expected: str) -> Path:
    found = sha256(path)
    if found != expected:
        path.unlink()
        raise ValueError(f"{path.name} has sha256 {found}, not the published {expected}: deleted it, run again")
    return path


def held(folder: Path) -> int:
    return sum(p.stat().st_size for p in folder.rglob("*") if p.is_file())


def appended(folder: Path, name: str) -> int:
    marker, partial = folder / f"{name}{JOINED}", folder / f"{name}{PARTIAL}"
    if not (marker.exists() and partial.exists()):
        marker.unlink(missing_ok=True)
        partial.unlink(missing_ok=True)
        return 0
    count, size = (int(x) for x in marker.read_text().split())
    os.truncate(partial, size)
    return count


def append(folder: Path, name: str, part: Path, count: int) -> None:
    partial, marker = folder / f"{name}{PARTIAL}", folder / f"{name}{JOINED}"
    with partial.open("ab") as out, part.open("rb") as source:
        shutil.copyfileobj(source, out, BUFFER)
        out.flush()
        os.fsync(out.fileno())
    fresh = folder / f"{name}{JOINED}.tmp"
    fresh.write_text(f"{count} {partial.stat().st_size}")
    os.replace(fresh, marker)
    part.unlink()


def joined(
    folder: Path, name: str, parts: list[str], published: dict[str, str], download: Callable[[str], Path], log: Log
) -> Path:
    final = folder / name
    if final.exists():
        return final
    folder.mkdir(parents=True, exist_ok=True)
    done = appended(folder, name)
    for count, part in enumerate(parts[done:], start=done + 1):
        log(f"part {count} of {len(parts)}: {part}")
        append(folder, name, checked(download(part), published[part]), count)
    log(f"checking {name} against its published checksum")
    os.replace(checked(folder / f"{name}{PARTIAL}", published[name]), final)
    (folder / f"{name}{JOINED}").unlink()
    return final


def report(folder: Path, total: int, log: Log, stop: threading.Event, every: float) -> None:
    while not stop.wait(every):
        log(f"progress: {held(folder) / GB:.1f} of {total / GB:.1f} GB on disk")


@contextlib.contextmanager
def reporting(folder: Path, total: int, log: Log, every: float = REPORT_SECONDS) -> Iterator[None]:
    stop = threading.Event()
    threading.Thread(target=report, args=(folder, total, log, stop, every), daemon=True).start()
    try:
        yield
    finally:
        stop.set()
