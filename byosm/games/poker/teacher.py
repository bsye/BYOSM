from __future__ import annotations

import subprocess
from collections.abc import Callable
from pathlib import Path

from ...config import STATE_DIR
from ...deploy import DOCKER_DIR, build_image, docker, docker_cli

REPO = "https://github.com/conorarmstrong/pluribus.git"
COMMIT = "757f7692738069522195d2b486eec60a8b010c0c"
IMAGE = f"byosm-poker-teacher:{COMMIT[:8]}"
DOCKERFILE = DOCKER_DIR / "poker-teacher.Dockerfile"
FOLDER = "poker-teacher"
CPUS = "16"
MEMORY = "28g"
TMPFS = "/tmp:size=1g"
HANDS = "/data/pluribus"
WORK = "/work"
CHECKPOINT = ".ckpt"
Log = Callable[[str], None]


def work_dir(ws: Path) -> Path:
    found = ws / STATE_DIR / FOLDER
    found.mkdir(parents=True, exist_ok=True)
    return found


def ensure_image(log: Log) -> None:
    if docker("image", "inspect", IMAGE).returncode != 0:
        build_image(DOCKERFILE, DOCKER_DIR, IMAGE, log, {"REPO": REPO, "COMMIT": COMMIT})


def command(ws: Path, hands: Path, args: list[str]) -> list[str]:
    sealed = [
        "--network",
        "none",
        "--read-only",
        "--tmpfs",
        TMPFS,
        "--cap-drop",
        "ALL",
        "--security-opt=no-new-privileges",
    ]
    limits = ["--cpus", CPUS, "--memory", MEMORY]
    mounts = ["-v", f"{work_dir(ws)}:{WORK}", "-v", f"{hands}:{HANDS}:ro"]
    return [docker_cli(), "run", "--rm", *sealed, *limits, *mounts, IMAGE, *args]


def run(ws: Path, hands: Path, args: list[str], log: Log) -> list[str]:
    ensure_image(log)
    said: list[str] = []
    with subprocess.Popen(command(ws, hands, args), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True) as p:
        for line in p.stdout or []:
            said.append(line.rstrip())
            log(f"  {said[-1]}")
    if p.returncode:
        raise RuntimeError(f"the poker teacher stopped with code {p.returncode}: {' '.join(args)}")
    return said


def blueprint(iters: int) -> str:
    return f"blueprint-{iters}.bin"


def train(ws: Path, hands: Path, iters: int, log: Log) -> Path:
    out = blueprint(iters)
    saved = f"{out}{CHECKPOINT}"
    resume = ["--resume", saved] if (work_dir(ws) / saved).exists() else []
    run(ws, hands, ["train", "--iters", str(iters), "--out", out, "--checkpoint", saved, *resume], log)
    return work_dir(ws) / out


def benchmark(ws: Path, hands: Path, iters: int, log: Log) -> list[str]:
    return run(ws, hands, ["benchmark", "--blueprint", blueprint(iters), "--dir", HANDS], log)
