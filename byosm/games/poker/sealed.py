from __future__ import annotations

import subprocess
from collections.abc import Callable
from pathlib import Path

from ...deploy import docker
from ...locks import exclusive

SEALED = (
    "--init",
    "--network",
    "none",
    "--read-only",
    "--tmpfs",
    "/tmp:size=1g",
    "--cap-drop",
    "ALL",
    "--security-opt=no-new-privileges",
)
SPARE_CPUS = 2
RESERVE_GIB = 3
GIB = 1024**3
Log = Callable[[str], None]


def limits() -> tuple[str, str]:
    found = docker("info", "--format", "{{.NCPU}} {{.MemTotal}}")
    if found.returncode:
        raise RuntimeError("a sealed poker container needs Docker running")
    cpus, memory = found.stdout.split()
    return str(max(1, int(cpus) - SPARE_CPUS)), f"{int(memory) // GIB - RESERVE_GIB}g"


def mounted(mounts: dict[Path, str], writable: tuple[str, ...] = ()) -> list[str]:
    flags: list[str] = []
    for source, target in mounts.items():
        if not source.suffix:
            source.mkdir(parents=True, exist_ok=True)
        flags += ["-v", f"{source}:{target}{'' if target in writable else ':ro'}"]
    return flags


def streamed(command: list[str], turn: Path, who: str, args: list[str], log: Log) -> list[str]:
    said: list[str] = []
    piped = {"stdout": subprocess.PIPE, "stderr": subprocess.STDOUT, "text": True}
    with (
        exclusive(turn, lambda: log("  waiting for the running poker container to finish")),
        subprocess.Popen(command, **piped) as p,
    ):
        for line in p.stdout or []:
            said.append(line.rstrip())
            log(f"  {said[-1]}")
    if p.returncode:
        raise RuntimeError(f"{who} stopped with code {p.returncode}: {' '.join(args)}")
    return said
