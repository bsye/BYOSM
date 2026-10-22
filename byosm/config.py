from __future__ import annotations

import json
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

if TYPE_CHECKING:
    from .spec import TaskSpec

CONFIG_NAME = "byosm.yaml"
ENV_FILE = ".env"
LLM_OVERRIDE_ENV = "BYOSM_LLM"
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


def find_workspace(start: Path | None = None) -> Path:
    p = (start or Path.cwd()).resolve()
    for d in [p, *p.parents]:
        if (d / CONFIG_NAME).exists():
            return d
    return p


def cache_root() -> Path:
    return Path(os.environ.get("BYOSM_CACHE") or Path.home() / ".cache" / "byosm")


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
    if os.environ.get(LLM_OVERRIDE_ENV):
        merged["llm"]["default"] = json.loads(os.environ[LLM_OVERRIDE_ENV])
    return merged


def write_default_config(ws: Path) -> Path:
    path = ws / CONFIG_NAME
    if not path.exists():
        path.write_text(yaml.safe_dump(DEFAULT_CONFIG, sort_keys=False))
    return path


class Project:
    def __init__(self, name: str, ws: Path | None = None):
        self.ws = ws or find_workspace()
        self.name = name
        self.root = self.ws / "specialists" / name

    @property
    def spec_path(self) -> Path:
        return self.root / "spec.yaml"

    def data_dir_for(self, spec: TaskSpec) -> Path:
        return self.root / "data" / spec.data_key()

    @property
    def data_dir(self) -> Path:
        from .spec import TaskSpec

        sp = self.best_dir / "spec.yaml"
        return self.data_dir_for(TaskSpec.load(sp if sp.exists() else self.spec_path))

    @property
    def reviewed_path(self) -> Path:
        return self.root / "reviewed.jsonl"

    @property
    def runs_dir(self) -> Path:
        return self.root / "experiments"

    def exp_dir(self, exp: str) -> Path:
        return self.runs_dir / exp

    def experiments(self) -> list[str]:
        if not self.runs_dir.exists():
            return []
        return sorted(p.name for p in self.runs_dir.glob("*") if (p / "meta.json").exists())

    @property
    def promoted(self) -> str | None:
        f = self.root / "promoted"
        return f.read_text().strip() if f.exists() else None

    def promote(self, exp: str) -> None:
        (self.root / "promoted").write_text(exp)

    @property
    def best_dir(self) -> Path:
        return self.exp_dir(self.promoted) if self.promoted else self.runs_dir / "__none__"

    @property
    def evals_dir(self) -> Path:
        return self.root / "evals"

    @property
    def baselines_dir(self) -> Path:
        return self.root / "baselines"

    def meta(self, exp: str) -> dict:
        f = self.exp_dir(exp) / "meta.json"
        return json.loads(f.read_text()) if f.exists() else {}

    @property
    def board_path(self) -> Path:
        return self.root / "board.json"

    @property
    def overfit_path(self) -> Path:
        return self.root / "overfit.json"

    def board(self) -> dict[str, Any]:
        return json.loads(self.board_path.read_text()) if self.board_path.exists() else {}

    @property
    def state_path(self) -> Path:
        return self.root / "state.json"

    def exists(self) -> bool:
        return self.spec_path.exists()

    def ensure(self) -> Project:
        for d in (self.root, self.runs_dir):
            d.mkdir(parents=True, exist_ok=True)
        return self


def free_name(base: str, ws: Path | None = None) -> str:
    taken = set(list_projects(ws))
    return next(n for n in (base, *(f"{base}-{i}" for i in range(2, len(taken) + 3))) if n not in taken)


def list_projects(ws: Path | None = None) -> list[str]:
    ws = ws or find_workspace()
    d = ws / "specialists"
    return sorted(p.name for p in d.iterdir() if (p / "spec.yaml").exists()) if d.exists() else []


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
