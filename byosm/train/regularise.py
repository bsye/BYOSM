from __future__ import annotations

import random
from collections.abc import Callable

import torch
from torch import nn

from .common import layer_depth, pretrained_encoders

LEAF = ".leaf"
INIT_STD = 0.02


def symmetric_kl(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    la, lb = nn.functional.log_softmax(a, -1), nn.functional.log_softmax(b, -1)
    one = nn.functional.kl_div(la, lb, log_target=True, reduction="batchmean")
    other = nn.functional.kl_div(lb, la, log_target=True, reduction="batchmean")
    return (one + other) / 2


def mixed_logits(
    model: nn.Module, inputs: dict[str, torch.Tensor], alpha: float, rng: random.Random
) -> tuple[torch.Tensor, list[int], float]:
    feats, mask = model.backbone(**inputs)
    head = next(iter(model.heads.values()))
    pooled = head.pooled(feats, mask)
    lam = rng.betavariate(alpha, alpha)
    perm = rng.sample(range(len(pooled)), len(pooled))
    return head.net(lam * pooled + (1 - lam) * pooled[perm]), perm, lam


def _reset(m: nn.Module, std: float) -> None:
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, std=std)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm) and m.elementwise_affine:
        nn.init.ones_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


@torch.no_grad()
def reinit_top(model: nn.Module, layers: int, log: Callable[[str], None]) -> int:
    done = 0
    for enc in pretrained_encoders(model.backbone).values():
        std = getattr(enc.config, "initializer_range", INIT_STD)
        named = [(m, layer_depth(name + LEAF)) for name, m in enc.named_modules()]
        top = max((d for _, d in named if d), default=0)
        picked = [(m, d) for m, d in named if d and d > top - layers]
        for m, _ in picked:
            _reset(m, std)
        done += len({d for _, d in picked})
    if done:
        log(f"  re-initialised the top {done} encoder layer(s)")
    return done
