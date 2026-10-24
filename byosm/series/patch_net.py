from __future__ import annotations

import numpy as np
import torch
from torch import nn

from .metrics import QUANTILES

CONTEXT = 512
PATCH = 16
WIDTH = 128
LAYERS = 3
HEADS = 4
MAX_HORIZON = 60


class PatchNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        patches = CONTEXT // PATCH
        self.embed = nn.Linear(2 * PATCH, WIDTH)
        self.position = nn.Parameter(torch.zeros(patches, WIDTH))
        layer = nn.TransformerEncoderLayer(WIDTH, HEADS, 4 * WIDTH, dropout=0.1, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, LAYERS)
        self.head = nn.Linear(patches * WIDTH, MAX_HORIZON * len(QUANTILES))

    def forward(self, values: torch.Tensor, seen: torch.Tensor) -> torch.Tensor:
        v, m = values.unfold(1, PATCH, PATCH), seen.unfold(1, PATCH, PATCH)
        x = self.encoder(self.embed(torch.cat([v, m], -1)) + self.position)
        return self.head(x.flatten(1)).view(-1, len(QUANTILES), MAX_HORIZON)


def window(history: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float]:
    ctx = history[-CONTEXT:]
    ctx = np.concatenate([np.full(CONTEXT - len(ctx), np.nan), ctx])
    seen = ~np.isnan(ctx)
    mean = float(np.mean(ctx[seen])) if seen.any() else 0.0
    std = float(np.std(ctx[seen])) if seen.any() else 1.0
    return np.where(seen, (ctx - mean) / (std or 1.0), 0.0), seen.astype(float), mean, std or 1.0


def pinball(pred: torch.Tensor, target: torch.Tensor, known: torch.Tensor) -> torch.Tensor:
    levels = torch.tensor(QUANTILES, device=pred.device).view(1, -1, 1)
    diff = target.unsqueeze(1) - pred
    loss = torch.maximum(levels * diff, (levels - 1) * diff) * known.unsqueeze(1)
    return loss.sum() / known.sum().clamp(min=1)


def batch(
    histories: list[np.ndarray], horizon: int, size: int, rng: np.random.Generator, device: str
) -> list[torch.Tensor]:
    values, seen, targets, known = [], [], [], []
    for i in rng.integers(0, len(histories), size):
        h = histories[i]
        end = int(rng.integers(PATCH, len(h) - horizon + 1))
        v, m, mean, std = window(h[:end])
        future = np.full(MAX_HORIZON, np.nan)
        future[:horizon] = (h[end : end + horizon] - mean) / std
        values.append(v)
        seen.append(m)
        targets.append(np.nan_to_num(future))
        known.append(~np.isnan(future))
    return [torch.tensor(np.asarray(a), dtype=torch.float32, device=device) for a in (values, seen, targets, known)]


def train(
    net: PatchNet, histories: list[np.ndarray], horizon: int, steps: int, lr: float, size: int, device: str, seed: int
) -> None:
    rng = np.random.default_rng(seed)
    usable = [h for h in histories if len(h) > horizon + PATCH]
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=steps)
    net.to(device).train()
    for _ in range(steps):
        values, seen, targets, known = batch(usable, horizon, size, rng, device)
        loss = pinball(net(values, seen), targets, known)
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        sched.step()
    net.to("cpu").eval()
