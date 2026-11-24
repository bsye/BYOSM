from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from ...config import STATE_DIR
from ...deploy import DOCKER_DIR, build_image, docker, docker_cli
from ...locks import exclusive

REPO = "https://github.com/conorarmstrong/pluribus.git"
COMMIT = "757f7692738069522195d2b486eec60a8b010c0c"
DOCKERFILE = DOCKER_DIR / "poker-teacher.Dockerfile"
PATCH = DOCKER_DIR / "poker-teacher.patch"
IMAGE = "byosm-poker-teacher"
FOLDER = "poker-teacher"
SPARE_CPUS = 2
RESERVE_GIB = 3
GIB = 1024**3
FREE_GIB = 40
TMPFS = "/tmp:size=1g"
HANDS = "/data/pluribus"
WORK = "/work"
STRATEGY = ".bin"
CHECKPOINT = ".bin.ckpt"
LABELS = ".labels.jsonl"
TURN = ".turn"
ITERS = 200_000_000
BUCKETS = 12
Log = Callable[[str], None]


@dataclass(frozen=True)
class Recipe:
    iters: int = ITERS
    buckets: int = BUCKETS
    snapshot_avg: bool = False
    linear_cap: int | None = None

    def name(self) -> str:
        buckets = [f"b{self.buckets}"] if self.buckets != BUCKETS else []
        snapshots = ["snap"] if self.snapshot_avg else []
        cap = [f"cap{self.linear_cap}"] if self.linear_cap else []
        return "-".join(["blueprint", str(self.iters), *buckets, *snapshots, *cap])

    def flags(self) -> list[str]:
        snapshots = ["--snapshot-avg"] if self.snapshot_avg else []
        cap = ["--linear-cap", str(self.linear_cap)] if self.linear_cap else []
        return ["--iters", str(self.iters), "--buckets", str(self.buckets), *snapshots, *cap]


DEFAULT = Recipe().name()


def image() -> str:
    build = hashlib.sha256(DOCKERFILE.read_bytes() + PATCH.read_bytes()).hexdigest()[:8]
    return f"{IMAGE}:{COMMIT[:8]}-{build}"


def work_dir(ws: Path) -> Path:
    found = ws / STATE_DIR / FOLDER
    found.mkdir(parents=True, exist_ok=True)
    return found


def ensure_image(log: Log) -> None:
    if docker("image", "inspect", image()).returncode != 0:
        build_image(DOCKERFILE, DOCKER_DIR, image(), log, {"REPO": REPO, "COMMIT": COMMIT})


def limits() -> tuple[str, str]:
    found = docker("info", "--format", "{{.NCPU}} {{.MemTotal}}")
    if found.returncode:
        raise RuntimeError("the poker teacher needs Docker running")
    cpus, memory = found.stdout.split()
    return str(max(1, int(cpus) - SPARE_CPUS)), f"{int(memory) // GIB - RESERVE_GIB}g"


def command(
    ws: Path,
    hands: Path | None,
    args: list[str],
    cpus: str | None = None,
    memory: str | None = None,
    flags: tuple[str, ...] = (),
) -> list[str]:
    sealed = [
        "--init",
        "--network",
        "none",
        "--read-only",
        "--tmpfs",
        TMPFS,
        "--cap-drop",
        "ALL",
        "--security-opt=no-new-privileges",
    ]
    most = limits() if cpus is None or memory is None else (cpus, memory)
    capped = ["--cpus", cpus or most[0], "--memory", memory or most[1]]
    mounts = ["-v", f"{work_dir(ws)}:{WORK}", *(["-v", f"{hands}:{HANDS}:ro"] if hands else [])]
    return [docker_cli(), "run", "--rm", *flags, *sealed, *capped, *mounts, image(), *args]


def run(ws: Path, hands: Path | None, args: list[str], log: Log) -> list[str]:
    ensure_image(log)
    said: list[str] = []
    piped = {"stdout": subprocess.PIPE, "stderr": subprocess.STDOUT, "text": True}
    with (
        exclusive(work_dir(ws) / TURN, lambda: log("  waiting for the running teacher container to finish")),
        subprocess.Popen(command(ws, hands, args), **piped) as p,
    ):
        for line in p.stdout or []:
            said.append(line.rstrip())
            log(f"  {said[-1]}")
    if p.returncode:
        raise RuntimeError(f"the poker teacher stopped with code {p.returncode}: {' '.join(args)}")
    return said


def room(ws: Path) -> None:
    free = shutil.disk_usage(work_dir(ws)).free // GIB
    if free < FREE_GIB:
        raise RuntimeError(
            f"only {free} GiB free on the disk holding {work_dir(ws)}: a teacher run writes checkpoints of 10 GiB "
            f"and more, so it needs {FREE_GIB} GiB. byosm poker teacher-remove NAME deletes a strategy's files"
        )


def remove(ws: Path, name: str) -> list[str]:
    found = sorted(p for p in work_dir(ws).iterdir() if p.name.startswith(f"{name}."))
    if not found:
        raise ValueError(f"nothing stored for a strategy named {name!r}")
    for path in found:
        path.unlink()
    return [p.name for p in found]


def strategy(ws: Path, name: str) -> str:
    if not (work_dir(ws) / f"{name}{STRATEGY}").exists():
        known = sorted(p.name.removesuffix(STRATEGY) for p in work_dir(ws).glob(f"*{STRATEGY}"))
        raise ValueError(f"no trained strategy named {name!r}; trained here: {', '.join(known) or 'none yet'}")
    return f"{name}{STRATEGY}"


def continued(ws: Path, name: str, start: str) -> None:
    source, target = (work_dir(ws) / f"{n}{CHECKPOINT}" for n in (start, name))
    if not source.exists():
        raise ValueError(f"no checkpoint for {start!r} to continue from")
    if not target.exists():
        os.link(source, target)


def train(ws: Path, hands: Path, recipe: Recipe, log: Log, start: str | None = None) -> str:
    name = recipe.name()
    saved = f"{name}{CHECKPOINT}"
    room(ws)
    if start:
        continued(ws, name, start)
    resume = ["--resume", saved] if (work_dir(ws) / saved).exists() else []
    run(ws, hands, ["train", *recipe.flags(), "--out", f"{name}{STRATEGY}", "--checkpoint", saved, *resume], log)
    return name


def benchmark(ws: Path, hands: Path, name: str, log: Log) -> list[str]:
    out = ["--out", f"{name}{LABELS}"]
    return run(ws, hands, ["benchmark", "--blueprint", strategy(ws, name), "--dir", HANDS, *out], log)
