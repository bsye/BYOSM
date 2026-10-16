"""The modality plugin contract. Adding a modality = one file implementing `Modality` + `@register`."""
from __future__ import annotations

import random
from typing import Any, Callable

import torch
from torch import nn

from ..spec import TaskSpec

IGNORE = -100


class Backbone(nn.Module):
    """forward(**inputs) -> (features [B, T, D], mask [B, T] bool). T=1 is fine for pooled encoders."""

    dim: int

    def forward(self, **inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:  # pragma: no cover
        raise NotImplementedError


class Modality:
    name: str = ""

    def __init__(self, spec: TaskSpec, data_dir: str | None = None):
        self.spec = spec
        self.data_dir = data_dir

    # -- data ------------------------------------------------------------------------------------
    def load(self, ex: dict) -> Any:
        """Record from data/*.jsonl -> raw input (str, waveform, PIL image...)."""
        raise NotImplementedError

    def frame_labels(self, ex: dict) -> list[int] | None:
        """For frame tasks: label index per native step (word / hop). None for classify."""
        return None

    # -- model -----------------------------------------------------------------------------------
    def build_backbone(self) -> Backbone:
        raise NotImplementedError

    def collate(self, raws: list[Any], frame_labels: list[list[int]] | None = None) -> tuple[dict[str, torch.Tensor], torch.Tensor | None]:
        """Batch raw inputs into model inputs. Frame targets come back padded with IGNORE; the model
        nearest-resamples them to its output length if the two differ (e.g. 10ms labels vs 20ms frames)."""
        raise NotImplementedError

    # -- self-supervised views (LeJEPA) ------------------------------------------------------------
    def views(self, raw: Any, rng: random.Random) -> tuple[list[Any], list[Any]]:
        """Return (global_views, local_views) of one input. Globals see most of the input, locals a small part."""
        raise NotImplementedError

    def example_input(self) -> Any:
        """A representative input for latency benchmarking."""
        raise NotImplementedError


REGISTRY: dict[str, type[Modality]] = {}


def register(cls: type[Modality]) -> type[Modality]:
    REGISTRY[cls.name] = cls
    return cls


def get_modality(spec: TaskSpec, data_dir: str | None = None) -> Modality:
    from . import audio, image, text  # noqa: F401  (populate registry)

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


Hook = Callable[..., Any]
