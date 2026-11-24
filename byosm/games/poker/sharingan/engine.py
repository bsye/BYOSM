from __future__ import annotations

import hashlib
from pathlib import Path

from ....deploy import DOCKER_DIR, build_image, docker
from ..sealed import Log
from ..teacher import TURN
from . import BUCKETS, CODE, COMMIT, DATA, WEIGHTS, Blueprint, work_dir

DOCKERFILE = DOCKER_DIR / "poker-sharingan.Dockerfile"
IMAGE = "byosm-poker-sharingan"
MOUNTED_WEIGHTS = "/weights"
MOUNTED_DATA = "/data"


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


def stored(ws: Path, found: Blueprint) -> None:
    missing = [p for p in (work_dir(ws) / WEIGHTS / found.file, work_dir(ws) / DATA / BUCKETS) if not p.exists()]
    if missing:
        raise ValueError(f"{missing[0]} is not here yet: run byosm poker blueprint-fetch first")


def turn(ws: Path) -> Path:
    return work_dir(ws) / TURN
