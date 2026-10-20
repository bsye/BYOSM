"""Workspace layout and settings (byosm.yaml)."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import yaml

CONFIG_NAME = "byosm.yaml"

DEFAULT_CONFIG: dict[str, Any] = {
    # Every LLM role falls back to `default`. Point any role at any provider:
    #   anthropic  -> Claude via the official SDK (ANTHROPIC_API_KEY or `ant auth login`)
    #   openai     -> any OpenAI-compatible endpoint: OpenAI, Ollama, vLLM, LM Studio, OpenRouter...
    "llm": {
        "default": {"provider": "anthropic", "model": "claude-opus-5-5", "effort": "medium"},
        # Bulk generation is the expensive part; a cheaper/faster model is usually fine.
        # "generate": {"provider": "openai", "model": "qwen3:14b", "base_url": "http://localhost:11434/v1", "api_key": "ollama"},
        # Use a *different* model family for the held-out eval set to avoid grading a model on its own generator's quirks.
        # "eval": {"provider": "openai", "model": "gpt-5", "api_key_env": "OPENAI_API_KEY"},
    },
    "concurrency": 8,
}


def find_workspace(start: Path | None = None) -> Path:
    """Walk up from `start` looking for byosm.yaml; fall back to cwd."""
    p = (start or Path.cwd()).resolve()
    for d in [p, *p.parents]:
        if (d / CONFIG_NAME).exists():
            return d
    return p


def load_config(ws: Path | None = None) -> dict[str, Any]:
    ws = ws or find_workspace()
    path = ws / CONFIG_NAME
    cfg = yaml.safe_load(path.read_text()) if path.exists() else {}
    merged = {**DEFAULT_CONFIG, **(cfg or {})}
    merged["llm"] = {**DEFAULT_CONFIG["llm"], **((cfg or {}).get("llm") or {})}
    return merged


def write_default_config(ws: Path) -> Path:
    path = ws / CONFIG_NAME
    if not path.exists():
        path.write_text(yaml.safe_dump(DEFAULT_CONFIG, sort_keys=False))
    return path


class Project:
    """One specialist's directory: specialists/<name>/{spec.yaml,data/,runs/,state.json}."""

    def __init__(self, name: str, ws: Path | None = None):
        self.ws = ws or find_workspace()
        self.name = name
        self.root = self.ws / "specialists" / name

    @property
    def spec_path(self) -> Path:
        return self.root / "spec.yaml"

    # Layout:
    #   spec.yaml                      base spec (what `byosm new` wrote; what you edit)
    #   data/<data_key>/               datasets, shared by every experiment with the same data plan
    #   experiments/<exp>/             spec.yaml snapshot, model.pt, meta.json, results/<suite>.json
    #   evals/<suite>/                 frozen eval suites (calib.jsonl, test.jsonl, media)
    #   baselines/<baseline>/          cached baseline results per suite
    #   promoted                       name of the experiment `predict`/`serve`/`export` use
    def data_dir_for(self, spec) -> Path:  # noqa: ANN001
        return self.root / "data" / spec.data_key()

    @property
    def data_dir(self) -> Path:
        """Data of the promoted experiment (or of the base spec)."""
        from .spec import TaskSpec

        sp = self.best_dir / "spec.yaml"
        return self.data_dir_for(TaskSpec.load(sp if sp.exists() else self.spec_path))

    @property
    def runs_dir(self) -> Path:
        return self.root / "experiments"

    def exp_dir(self, exp: str) -> Path:
        return self.runs_dir / exp

    @property
    def board_path(self) -> Path:
        return self.root / "board.json"

    def board(self) -> dict:
        return json.loads(self.board_path.read_text()) if self.board_path.exists() else {}

    def experiments(self) -> list[str]:
        return sorted(p.name for p in self.runs_dir.glob("*") if (p / "meta.json").exists()) if self.runs_dir.exists() else []

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

    @property
    def state_path(self) -> Path:
        return self.root / "state.json"

    def exists(self) -> bool:
        return self.spec_path.exists()

    def ensure(self) -> "Project":
        for d in (self.root, self.runs_dir):
            d.mkdir(parents=True, exist_ok=True)
        return self


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
