from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from ..deploy import docker

NVIDIA = "nvidia"
NVIDIA_HOOK = "nvidia-container-runtime-hook"
DESKTOP = "Docker Desktop"
HOST_PROVIDERS = ("claude-code", "codex")
APPLE_GPU = "the job trains on the Apple GPU, which Docker cannot reach"
NOT_LINUX = "Docker runs containers in a VM here, and the workspace database's file locks do not cross into it"
NO_DOCKER = "Docker is not installed or its daemon does not answer"
NO_NVIDIA = "the job trains on CUDA and Docker has no NVIDIA runtime"
DATABASE_SERVER = "the workspace database is a server outside the container's network"
HOST_LLM = "the job's LLM is a command-line tool on this machine"


def daemon() -> dict[str, Any] | None:
    if not shutil.which("docker"):
        return None
    try:
        out = docker("info", "--format", "{{json .}}")
        return json.loads(out.stdout) if out.returncode == 0 else None
    except (subprocess.TimeoutExpired, ValueError):
        return None


def _nvidia(info: dict[str, Any]) -> bool:
    return NVIDIA in (info.get("Runtimes") or {}) or bool(shutil.which(NVIDIA_HOOK))


def blocker(ws: Path, llms: list[dict[str, Any]], device: str) -> str | None:
    from ..db.engine import database_url

    if device == "mps":
        return APPLE_GPU
    if not sys.platform.startswith("linux"):
        return NOT_LINUX
    info = daemon()
    if info is None:
        return NO_DOCKER
    if DESKTOP in str(info.get("OperatingSystem")):
        return NOT_LINUX
    if device == "cuda" and not _nvidia(info):
        return NO_NVIDIA
    if not database_url(ws).startswith(f"sqlite:///{ws}"):
        return DATABASE_SERVER
    if any(conf.get("provider") in HOST_PROVIDERS for conf in llms):
        return HOST_LLM
    return None
