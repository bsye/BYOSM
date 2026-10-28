from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from ..app.detached import wrapped
from ..app.job import Job, job_vars
from ..config import cache_root, env_file
from .containers import HARDENED, JOB_PREFIX, LABEL, NETWORK_PREFIX, SCRATCH, proxy_url

SHM_SIZE = "2g"
PIDS_LIMIT = 16384
MB_PER_GB = 1024
CONTAINER_USER = "byosm"
CONTAINER_PYTHON = "python"
ALL_GPUS, NO_GPUS = "all", "none"
PROXY_VARS = ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy")
PASSED_PREFIXES = ("BYOSM_", "HF_")


def hf_home() -> Path:
    from huggingface_hub import constants

    return Path(constants.HF_HOME)


def gpu_flags(gpus: str) -> list[str]:
    if gpus == NO_GPUS:
        return []
    return ["--gpus", gpus if gpus == ALL_GPUS or gpus.isdigit() else f'"device={gpus}"']


def _mounts(ws: Path) -> list[str]:
    folders = [ws, cache_root(), hf_home()]
    for folder in folders:
        folder.mkdir(parents=True, exist_ok=True)
    return [f"--volume={folder}:{folder}" for folder in folders]


def _fixed(job: Job) -> dict[str, str]:
    paths = {"HOME": SCRATCH, "USER": CONTAINER_USER, "BYOSM_CACHE": str(cache_root()), "HF_HOME": str(hf_home())}
    return {**paths, **dict.fromkeys(PROXY_VARS, proxy_url(job.id))}


def passed_names(ws: Path, job: Job, fixed: dict[str, str]) -> list[str]:
    from ..app.keys import KNOWN_KEYS
    from ..db.engine import URL_ENV

    host = [n for n in os.environ if n.startswith(PASSED_PREFIXES) or n in (*KNOWN_KEYS, URL_ENV)]
    return sorted({*job_vars(job), *env_file(ws), *host} - set(fixed))


def _env(ws: Path, job: Job) -> list[str]:
    fixed = _fixed(job)
    return [f"--env={k}={v}" for k, v in fixed.items()] + [f"--env={n}" for n in passed_names(ws, job, fixed)]


def run_args(
    ws: Path, job: Job, entry: str, image: str, limits: dict[str, Any], gpus: list[str], inner: Path
) -> list[str]:
    name = ["--init", "--log-driver", "none", "--name", JOB_PREFIX + job.id, "--label", f"{LABEL}={job.id}"]
    memory = f"{int(limits['memory_gb'] * MB_PER_GB)}m"
    sizes = ["--cpus", str(limits["cpus"]), "--memory", memory, "--shm-size", SHM_SIZE, "--pids-limit", str(PIDS_LIMIT)]
    place = ["--network", NETWORK_PREFIX + job.id, "--user", f"{os.getuid()}:{os.getgid()}", "--workdir", str(ws)]
    command = wrapped(inner, entry, job.args, CONTAINER_PYTHON)
    return [*name, *sizes, *HARDENED, *place, *_mounts(ws), *gpus, *_env(ws, job), image, *command]
