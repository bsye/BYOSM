from __future__ import annotations

import importlib
import random
from collections.abc import Callable
from typing import Any

import torch
from torch import nn

from ..spec import TaskSpec

IGNORE = -100
_PLUGINS = ("audio", "document", "image", "text")


class Backbone(nn.Module):
    dim: int

    def forward(self, **inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError


class Modality:
    name: str = ""

    def __init__(self, spec: TaskSpec, data_dir: str | None = None):
        self.spec = spec
        self.data_dir = data_dir

    def load(self, ex: dict) -> Any:
        raise NotImplementedError

    def load_many(self, rows: list[dict]) -> list[Any]:
        return [self.load(r) for r in rows]

    def frame_labels(self, ex: dict) -> list[int] | None:
        return None

    def frame_label_ids(self, ex: dict, numeric_ok: bool = True) -> list[int] | None:
        if self.spec.task_type != "frame":
            return None
        idx = self.spec.label_index
        return [idx[lab] if isinstance(lab, str) or not numeric_ok else int(lab) for lab in ex["label"]]

    def build_backbone(self) -> Backbone:
        raise NotImplementedError

    def collate(
        self, raws: list[Any], frame_labels: list[list[int]] | None = None
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor | None]:
        raise NotImplementedError

    def views(self, raw: Any, rng: random.Random) -> tuple[list[Any], list[Any]]:
        raise NotImplementedError

    def example_input(self) -> Any:
        raise NotImplementedError


REGISTRY: dict[str, type[Modality]] = {}


def register(cls: type[Modality]) -> type[Modality]:
    REGISTRY[cls.name] = cls
    return cls


def get_modality(spec: TaskSpec, data_dir: str | None = None) -> Modality:
    for plugin in _PLUGINS:
        importlib.import_module(f"{__package__}.{plugin}")
    if spec.modality not in REGISTRY:
        raise KeyError(f"no modality plugin {spec.modality!r}; known: {sorted(REGISTRY)}")
    return REGISTRY[spec.modality](spec, data_dir)


def pad_labels(seqs: list[list[int]], length: int | None = None) -> torch.Tensor:
    n = length or max((len(s) for s in seqs), default=1)
    out = torch.full((len(seqs), n), IGNORE, dtype=torch.long)
    for i, s in enumerate(seqs):
        s = s[:n]
        out[i, : len(s)] = torch.tensor(s, dtype=torch.long)
    return out


def masked_mean(feats: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    m = mask.unsqueeze(-1).to(feats.dtype)
    return (feats * m).sum(1) / m.sum(1).clamp(min=1.0)


TORCH_WEIGHTS = (
    "model.safetensors",
    "pytorch_model.bin",
    "model.safetensors.index.json",
    "pytorch_model.bin.index.json",
)
Hook = Callable[..., Any]


def trainable(repo: str) -> bool:
    from huggingface_hub import list_repo_files

    try:
        files = list_repo_files(repo)
    except Exception:
        return True
    return any(name.rpartition("/")[2] in TORCH_WEIGHTS for name in files)


def pretrained(repo: str) -> nn.Module:
    from transformers import AutoModel

    return AutoModel.from_pretrained(repo, dtype=torch.float32)
