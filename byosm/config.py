from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

if TYPE_CHECKING:
    pass

STATE_DIR = Path(".byosm")
CONFIG_NAME = "byosm.yaml"
ENV_FILE = ".env"
LLM_OVERRIDE_ENV = "BYOSM_LLM"
CONCURRENCY_ENV = "BYOSM_CONCURRENCY"
TESTSET_PER_LABEL = 20
TESTSET_SUITE = "llm-written"
HF_TOKEN_KEY = "HF_TOKEN"

DEFAULT_CONFIG: dict[str, Any] = {
    "llm": {
        "default": {"provider": "anthropic", "model": "claude-opus-5-5", "effort": "medium"},
    },
    "concurrency": 8,
}


PUBLISH_LICENSE = "cc-by-nc-4.0"
JOBS_KEY = "jobs"
PROCESS, CONTAINER = "process", "container"
JOB_DEFAULTS: dict[str, Any] = {
    "isolation": PROCESS,
    "cpus": None,
    "memory_gb": None,
    "gpus": "all",
    "nice": 10,
    "cpu_hours": None,
    "network": [],
}


def find_workspace(start: Path | None = None) -> Path:
    p = (start or Path.cwd()).resolve()
    for d in [p, *p.parents]:
        if (d / CONFIG_NAME).exists():
            return d
    return p


def cache_root() -> Path:
    return Path(os.environ.get("BYOSM_CACHE") or Path.home() / ".cache" / "byosm")


def cache_db(file: str, table: str) -> sqlite3.Connection:
    cache_root().mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(cache_root() / file)
    db.execute(f"create table if not exists {table}")
    return db


def env_file(ws: Path | None = None) -> dict[str, str]:
    env = (ws or find_workspace()) / ENV_FILE
    lines = env.read_text().splitlines() if env.exists() else []
    pairs = (ln.partition("=") for ln in lines if "=" in ln and not ln.lstrip().startswith("#"))
    return {k.strip(): v.strip().strip("'\"") for k, _, v in pairs}


def secret(name: str, ws: Path | None = None) -> str | None:
    return os.environ.get(name) or env_file(ws).get(name)


def stored_config(ws: Path) -> dict[str, Any]:
    path = ws / CONFIG_NAME
    return (yaml.safe_load(path.read_text()) or {}) if path.exists() else {}


def save_config(ws: Path, cfg: dict[str, Any]) -> None:
    (ws / CONFIG_NAME).write_text(yaml.safe_dump(cfg, sort_keys=False))


def load_config(ws: Path | None = None) -> dict[str, Any]:
    ws = ws or find_workspace()
    cfg = stored_config(ws)
    merged = {**DEFAULT_CONFIG, **(cfg or {})}
    merged["llm"] = {**DEFAULT_CONFIG["llm"], **((cfg or {}).get("llm") or {})}
    if os.environ.get(CONCURRENCY_ENV):
        merged["concurrency"] = int(os.environ[CONCURRENCY_ENV])
    if os.environ.get(LLM_OVERRIDE_ENV):
        merged["llm"] = {"default": json.loads(os.environ[LLM_OVERRIDE_ENV])}
    return merged


def job_settings(ws: Path | None = None) -> dict[str, Any]:
    settings = {**JOB_DEFAULTS, **(load_config(ws).get(JOBS_KEY) or {})}
    if settings["isolation"] not in (PROCESS, CONTAINER):
        raise ValueError(f"{JOBS_KEY}.isolation must be {PROCESS} or {CONTAINER}, not {settings['isolation']!r}")
    return settings


def write_default_config(ws: Path) -> Path:
    path = ws / CONFIG_NAME
    if not path.exists():
        path.write_text(yaml.safe_dump(DEFAULT_CONFIG, sort_keys=False))
    return path


def device() -> str:
    import torch

    forced = os.environ.get("BYOSM_DEVICE")
    if forced:
        return forced
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"
