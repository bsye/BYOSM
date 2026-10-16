"""Backbone + tiny task head(s) + LeJEPA projector."""
from __future__ import annotations

import torch
from torch import nn

from .modalities import IGNORE, Backbone, masked_mean


class Head(nn.Module):
    def __init__(self, dim: int, n_labels: int, task_type: str):
        super().__init__()
        self.task_type = task_type
        self.net = nn.Sequential(nn.LayerNorm(dim), nn.Dropout(0.1), nn.Linear(dim, n_labels))

    def forward(self, feats: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if self.task_type == "frame":
            return self.net(feats)  # [B, T, C]
        return self.net(masked_mean(feats, mask))  # [B, C]


class SpecialistModel(nn.Module):
    """One backbone, one or more heads. A specialist has exactly one head; the `bench` generalist has one per task."""

    def __init__(self, backbone: Backbone, heads: dict[str, tuple[int, str]], proj_dim: int = 128):
        super().__init__()
        self.backbone = backbone
        d = backbone.dim
        self.heads = nn.ModuleDict({k: Head(d, n, t) for k, (n, t) in heads.items()})
        self.proj = nn.Sequential(nn.Linear(d, 512), nn.BatchNorm1d(512), nn.GELU(), nn.Linear(512, proj_dim))

    def embed(self, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        feats, mask = self.backbone(**inputs)
        return self.proj(masked_mean(feats, mask))

    def forward(self, inputs: dict[str, torch.Tensor], head: str | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        feats, mask = self.backbone(**inputs)
        h = self.heads[head or next(iter(self.heads))]
        return h(feats, mask), mask


def align_targets(targets: torch.Tensor, n: int) -> torch.Tensor:
    """Nearest-resample frame targets [B, T0] to the model's output length n."""
    if targets.shape[1] == n:
        return targets
    idx = (torch.arange(n, device=targets.device).float() * targets.shape[1] / n).long().clamp(max=targets.shape[1] - 1)
    return targets[:, idx]


def task_loss(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor, weight: torch.Tensor | None = None) -> torch.Tensor:
    if logits.dim() == 3:
        t = align_targets(targets, logits.shape[1]).clone()
        t[~mask] = IGNORE
        return nn.functional.cross_entropy(logits.flatten(0, 1), t.flatten(), weight=weight, ignore_index=IGNORE, label_smoothing=0.02)
    return nn.functional.cross_entropy(logits, targets, weight=weight, label_smoothing=0.02)


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())
