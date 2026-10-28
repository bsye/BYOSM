from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from ..lab.contract import Hooks
from ..train.gpu_memory import release_cache
from . import checkpoint
from .augment import augmented, hidden
from .metrics import QUANTILES
from .recipe import V1, Recipe

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
LONG = Size(context=2048, patch=32, width=192, layers=6, heads=6)
SIZES = {"base": BASE, "large": LARGE, "long": LONG}


class PatchNet(nn.Module):
    def __init__(self, size: Size = SMALL, levels: int = len(QUANTILES)) -> None:
        super().__init__()
        self.size = size
        patches = size.context // size.patch
        self.embed = nn.Linear(2 * size.patch, size.width)
        self.position = nn.Parameter(torch.zeros(patches, size.width))
        layer = nn.TransformerEncoderLayer(
            size.width, size.heads, 4 * size.width, dropout=0.1, batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, size.layers)
        self.head = nn.Linear(patches * size.width, MAX_HORIZON * levels)

    def forward(self, values: torch.Tensor, seen: torch.Tensor) -> torch.Tensor:
        p = self.size.patch
        v, m = values.unfold(1, p, p), seen.unfold(1, p, p)
        x = self.encoder(self.embed(torch.cat([v, m], -1)) + self.position)
        return self.head(x.flatten(1)).view(-1, self.head.out_features // MAX_HORIZON, MAX_HORIZON)


def window(history: np.ndarray, context: int) -> tuple[np.ndarray, np.ndarray, float, float]:
    ctx = history[-context:]
    ctx = np.concatenate([np.full(context - len(ctx), np.nan), ctx])
    seen = ~np.isnan(ctx)
    mean = float(np.mean(ctx[seen])) if seen.any() else 0.0
    std = float(np.std(ctx[seen])) if seen.any() else 1.0
    std = max(std, MIN_SCALE * abs(mean)) or 1.0
    return np.where(seen, (ctx - mean) / std, 0.0), seen.astype(float), mean, std


def pinball(
    pred: torch.Tensor, target: torch.Tensor, known: torch.Tensor, quantiles: tuple[float, ...] = QUANTILES
) -> torch.Tensor:
    levels = torch.tensor(quantiles, device=pred.device).view(1, -1, 1)
    diff = target.unsqueeze(1) - pred
    loss = torch.maximum(levels * diff, (levels - 1) * diff) * known.unsqueeze(1)
    return loss.sum() / known.sum().clamp(min=1)


def _loss(
    pred: torch.Tensor, target: torch.Tensor, known: torch.Tensor, step: int, recipe: Recipe, hooks: Hooks
) -> torch.Tensor:
    if "loss" not in hooks:
        return pinball(pred, target, known, recipe.quantiles)
    levels = torch.tensor(recipe.quantiles, device=pred.device)
    return hooks["loss"](pred, target, known, levels, step)


def reshaped(segments: list[np.ndarray], context: int, rng: np.random.Generator, hooks: Hooks) -> list[np.ndarray]:
    if "window" not in hooks:
        return segments
    out = [np.asarray(hooks["window"](s.copy(), context, rng), dtype=float) for s in segments]
    if any(o.shape != s.shape for o, s in zip(out, segments, strict=True)):
        raise ValueError(f"the window hook must return {len(segments[0])} values, as many as it was given")
    return out


def picker(series: list[tuple[np.ndarray, int]], patch: int) -> Pick:
    usable = [(h, horizon) for h, horizon in series if len(h) > horizon + patch]
    return lambda rng, n: [usable[i] for i in rng.integers(0, len(usable), n)]


def _segment(h: np.ndarray, horizon: int, size: Size, rng: np.random.Generator) -> np.ndarray:
    end = int(rng.integers(size.patch, len(h) - horizon + 1))
    past = h[max(0, end - size.context) : end]
    out = np.full(size.context + MAX_HORIZON, np.nan)
    out[size.context - len(past) : size.context] = past
    out[size.context : size.context + horizon] = h[end : end + horizon]
    return out


def batch(
    chosen: list[tuple[np.ndarray, int]],
    size: Size,
    rng: np.random.Generator,
    device: str,
    recipe: Recipe = V1,
    hooks: Hooks | None = None,
) -> list[torch.Tensor]:
    segments = augmented([_segment(h, horizon, size, rng) for h, horizon in chosen], size.context, recipe, rng)
    values, seen, targets, known = [], [], [], []
    for segment in reshaped(segments, size.context, rng, hooks or {}):
        past = hidden(segment[: size.context], size.patch, recipe.mask_rate, rng)
        v, m, mean, std = window(past, size.context)
        future = recipe.scale(np.clip((segment[size.context :] - mean) / std, -TARGET_CAP, TARGET_CAP))
        values.append(recipe.scale(v))
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
    resume: Path | None = None,
    recipe: Recipe = V1,
    hooks: Hooks | None = None,
) -> None:
    hooks = hooks or {}
    rng = np.random.default_rng(seed)
    net.to(device).train()
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=steps)
    done = checkpoint.restore(resume, net, opt, sched, rng, device) if resume else 0
    if done:
        print(f"  resuming from step {done}/{steps}")
    for step in range(done + 1, steps + 1):
        values, seen, targets, known = batch(pick(rng, size), net.size, rng, device, recipe, hooks)
        loss = _loss(net(values, seen), targets, known, step - 1, recipe, hooks)
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        sched.step()
        if "after_step" in hooks:
            hooks["after_step"](net, step - 1, steps)
        if report and step % REPORT_EVERY == 0:
            report(step, loss.item())
        if resume and step % checkpoint.EVERY == 0 and step < steps:
            checkpoint.save(resume, step, net, opt, sched, rng)
    if resume:
        resume.unlink(missing_ok=True)
    net.to("cpu").eval()
    release_cache()
