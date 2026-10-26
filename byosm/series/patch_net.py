from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from ..train.gpu_memory import release_cache
from .metrics import QUANTILES

MAX_HORIZON = 60
REPORT_EVERY = 500
MIN_SCALE = 0.01
TARGET_CAP = 50.0
Pick = Callable[[np.random.Generator, int], list[tuple[np.ndarray, int]]]


@dataclass(frozen=True)
class Size:
    context: int
    patch: int
    width: int
    layers: int
    heads: int


SMALL = Size(context=512, patch=16, width=128, layers=3, heads=4)
BASE = Size(context=1024, patch=32, width=192, layers=6, heads=6)
LARGE = Size(context=1024, patch=32, width=384, layers=8, heads=8)
SIZES = {"base": BASE, "large": LARGE}


class PatchNet(nn.Module):
    def __init__(self, size: Size = SMALL) -> None:
        super().__init__()
        self.size = size
        patches = size.context // size.patch
        self.embed = nn.Linear(2 * size.patch, size.width)
        self.position = nn.Parameter(torch.zeros(patches, size.width))
        layer = nn.TransformerEncoderLayer(
            size.width, size.heads, 4 * size.width, dropout=0.1, batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, size.layers)
        self.head = nn.Linear(patches * size.width, MAX_HORIZON * len(QUANTILES))

    def forward(self, values: torch.Tensor, seen: torch.Tensor) -> torch.Tensor:
        p = self.size.patch
        v, m = values.unfold(1, p, p), seen.unfold(1, p, p)
        x = self.encoder(self.embed(torch.cat([v, m], -1)) + self.position)
        return self.head(x.flatten(1)).view(-1, len(QUANTILES), MAX_HORIZON)


def window(history: np.ndarray, context: int) -> tuple[np.ndarray, np.ndarray, float, float]:
    ctx = history[-context:]
    ctx = np.concatenate([np.full(context - len(ctx), np.nan), ctx])
    seen = ~np.isnan(ctx)
    mean = float(np.mean(ctx[seen])) if seen.any() else 0.0
    std = float(np.std(ctx[seen])) if seen.any() else 1.0
    std = max(std, MIN_SCALE * abs(mean)) or 1.0
    return np.where(seen, (ctx - mean) / std, 0.0), seen.astype(float), mean, std


def pinball(pred: torch.Tensor, target: torch.Tensor, known: torch.Tensor) -> torch.Tensor:
    levels = torch.tensor(QUANTILES, device=pred.device).view(1, -1, 1)
    diff = target.unsqueeze(1) - pred
    loss = torch.maximum(levels * diff, (levels - 1) * diff) * known.unsqueeze(1)
    return loss.sum() / known.sum().clamp(min=1)


def picker(series: list[tuple[np.ndarray, int]], patch: int) -> Pick:
    usable = [(h, horizon) for h, horizon in series if len(h) > horizon + patch]
    return lambda rng, n: [usable[i] for i in rng.integers(0, len(usable), n)]


def batch(
    chosen: list[tuple[np.ndarray, int]], size: Size, rng: np.random.Generator, device: str
) -> list[torch.Tensor]:
    values, seen, targets, known = [], [], [], []
    for h, horizon in chosen:
        end = int(rng.integers(size.patch, len(h) - horizon + 1))
        v, m, mean, std = window(h[:end], size.context)
        future = np.full(MAX_HORIZON, np.nan)
        future[:horizon] = np.clip((h[end : end + horizon] - mean) / std, -TARGET_CAP, TARGET_CAP)
        values.append(v)
        seen.append(m)
        targets.append(np.nan_to_num(future))
        known.append(~np.isnan(future))
    return [torch.tensor(np.asarray(a), dtype=torch.float32, device=device) for a in (values, seen, targets, known)]


def train(
    net: PatchNet,
    pick: Pick,
    steps: int,
    lr: float,
    size: int,
    device: str,
    seed: int,
    report: Callable[[int, float], None] | None = None,
) -> None:
    rng = np.random.default_rng(seed)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=steps)
    net.to(device).train()
    for step in range(1, steps + 1):
        values, seen, targets, known = batch(pick(rng, size), net.size, rng, device)
        loss = pinball(net(values, seen), targets, known)
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        sched.step()
        if report and step % REPORT_EVERY == 0:
            report(step, loss.item())
    net.to("cpu").eval()
    release_cache()
