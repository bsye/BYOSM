from __future__ import annotations

import math
import random
from collections import defaultdict
from collections.abc import Callable

import numpy as np
import torch

from ..modalities import Modality
from ..model import SpecialistModel, warmup_cosine
from ..spec import TaskSpec
from .common import head_of, to_device

MAX_PAIRS = 20_000
WARMUP_SHARE = 0.1
SAME, DIFFERENT = 1.0, 0.0

Pair = tuple[int, int, float]


def _partner(group: list[int], i: int, rng: random.Random) -> int:
    j = rng.choice(group)
    while j == i:
        j = rng.choice(group)
    return j


def pairs(labels: list[int], per_row: int, rng: random.Random) -> list[Pair]:
    by_label: dict[int, list[int]] = defaultdict(list)
    for i, y in enumerate(labels):
        by_label[y].append(i)
    others = {y: [j for j, z in enumerate(labels) if z != y] for y in by_label}
    half = max(1, min(per_row, MAX_PAIRS // max(1, len(labels))) // 2)
    out: list[Pair] = []
    for i, y in enumerate(labels):
        if len(by_label[y]) > 1:
            out += [(i, _partner(by_label[y], i, rng), SAME) for _ in range(half)]
        if others[y]:
            out += [(i, rng.choice(others[y]), DIFFERENT) for _ in range(half)]
    rng.shuffle(out)
    return out


def _pair_loss(model: SpecialistModel, mod: Modality, raws: list[str], chunk: list[Pair], dev: str) -> torch.Tensor:
    inputs, _ = mod.collate([raws[a] for a, _, _ in chunk] + [raws[b] for _, b, _ in chunk])
    feats, mask = model.backbone(**to_device(inputs, dev))
    u, v = head_of(model).pooled(feats, mask).split(len(chunk))
    target = torch.tensor([t for _, _, t in chunk], device=dev)
    return torch.nn.functional.mse_loss(torch.cosine_similarity(u, v), target)


def contrastive(
    spec: TaskSpec,
    mod: Modality,
    model: SpecialistModel,
    raws: list[str],
    labels: list[int],
    rng: random.Random,
    dev: str,
    log: Callable[[str], None],
) -> list[float]:
    tc = spec.train
    todo = pairs(labels, tc.pairs_per_row, rng)
    params = list(model.backbone.parameters())
    opt = torch.optim.AdamW(params, lr=tc.lr_pairs, weight_decay=0.01)
    steps = max(1, math.ceil(len(todo) / tc.batch_size))
    sched = warmup_cosine(opt, steps, int(steps * WARMUP_SHARE))
    model.train()
    losses = []
    for start in range(0, len(todo), tc.batch_size):
        loss = _pair_loss(model, mod, raws, todo[start : start + tc.batch_size], dev)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        losses.append(loss.item())
    if losses:
        log(f"  contrastive pairs: {len(todo)}, loss {losses[0]:.4f} -> {np.mean(losses[-10:]):.4f}")
    return losses
