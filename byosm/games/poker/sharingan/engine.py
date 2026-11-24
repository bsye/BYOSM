from __future__ import annotations

import hashlib
from pathlib import Path

from ....deploy import DOCKER_DIR, build_image, docker, docker_cli
from ..sealed import SEALED, Log, limits, streamed
from ..teacher import TURN
from ..teacher import work_dir as teacher_dir
from . import BUCKETS, CODE, COMMIT, DATA, WEIGHTS, Blueprint, work_dir

DOCKERFILE = DOCKER_DIR / "poker-sharingan.Dockerfile"
IMAGE = "byosm-poker-sharingan"
MOUNTED_WEIGHTS = "/weights"
MOUNTED_DATA = "/data"
BENCH = "bench"


def image() -> str:
    build = hashlib.sha256(DOCKERFILE.read_bytes()).hexdigest()[:8]
    return f"{IMAGE}:{COMMIT[:8]}-{build}"


def ensure_image(log: Log) -> None:
    if docker("image", "inspect", image()).returncode != 0:
        build_image(DOCKERFILE, DOCKER_DIR, image(), log, {"REPO": CODE, "COMMIT": COMMIT})


def policy(found: Blueprint) -> str:
    return f"compact:{MOUNTED_WEIGHTS}/{found.file}"


def loading(found: Blueprint) -> list[str]:
    return ["--bucketer", f"tables:{MOUNTED_DATA}/{BUCKETS}", "--sizing", found.sizing]


def command(ws: Path, args: list[str]) -> list[str]:
    cpus, memory = limits()
    root = work_dir(ws)
    mounts = ["-v", f"{root / WEIGHTS}:{MOUNTED_WEIGHTS}:ro", "-v", f"{root / DATA}:{MOUNTED_DATA}:ro"]
    return [docker_cli(), "run", "--rm", *SEALED, "--cpus", cpus, "--memory", memory, *mounts, image(), *args]


def stored(ws: Path, found: Blueprint) -> None:
    missing = [p for p in (work_dir(ws) / WEIGHTS / found.file, work_dir(ws) / DATA / BUCKETS) if not p.exists()]
    if missing:
        raise ValueError(f"{missing[0]} is not here yet: run byosm poker blueprint-fetch first")


def run(ws: Path, args: list[str], log: Log) -> list[str]:
    ensure_image(log)
    return streamed(command(ws, args), teacher_dir(ws) / TURN, "the SHARINGAN container", args, log)
