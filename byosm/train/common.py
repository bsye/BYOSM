from __future__ import annotations

import contextlib
import copy
import functools
import math
import random
from typing import Any

import torch

from ..modalities import Modality
from ..model import SpecialistModel
from ..spec import TaskSpec

LAYER_SEGMENTS = {"layer", "layers", "blocks", "block", "h", "stages"}


def targets_of(spec: TaskSpec, mod: Modality, rows: list[dict]) -> list[Any]:
    if spec.task_type == "frame":
        return [mod.frame_labels(r) for r in rows]
    idx = {n: i for i, n in enumerate(spec.label_names)}
    return [idx[r["label"]] for r in rows]


def _lr_factor(s: int, warm: int, steps: int) -> float:
    return min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(s, steps) / steps))


def warmup_cosine(opt: torch.optim.Optimizer, steps: int) -> torch.optim.lr_scheduler.LambdaLR:
    lr_lambda = functools.partial(_lr_factor, warm=max(1, steps // 15), steps=steps)
    return torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)


def cpu_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return copy.deepcopy({k: v.cpu() for k, v in model.state_dict().items()})


def batches(n: int, bs: int, shuffle: bool, rng: random.Random) -> list[list[int]]:
    order = list(range(n))
    if shuffle:
        rng.shuffle(order)
    return [order[i : i + bs] for i in range(0, n, bs)]


def to_device(inputs: dict[str, torch.Tensor], dev: str) -> dict[str, torch.Tensor]:
    return {k: v.to(dev) for k, v in inputs.items()}


def build_model(spec: TaskSpec, mod: Modality) -> SpecialistModel:
    return SpecialistModel(mod.build_backbone(), {spec.name: (len(spec.labels), spec.task_type, spec.train.pool)})


def pretrained_encoders(model: torch.nn.Module) -> dict[str, torch.nn.Module]:
    from transformers import PreTrainedModel

    out: dict[str, torch.nn.Module] = {}
    for name, m in model.named_modules():
        if isinstance(m, PreTrainedModel) and not any(name.startswith(p + ".") for p in out):
            out[name] = m
    return out


def layer_depth(name: str) -> int | None:
    parts = name.split(".")
    for segment, index in zip(parts, parts[1:-1], strict=False):
        if segment in LAYER_SEGMENTS and index.isdecimal():
            return int(index) + 1
    return 0 if "embed" in name else None


def _encoder_groups(m: torch.nn.Module, tc: Any, groups: dict[float, list], seen: set[int]) -> None:
    named = list(m.named_parameters())
    depth = {n: layer_depth(n) for n, _ in named}
    top = max([d for d in depth.values() if d is not None], default=0) + 1
    decay = tc.layer_decay_floor ** (1 / top) if top > 0 else 1.0
    for n, p in named:
        d = top if depth[n] is None else depth[n]
        groups.setdefault(tc.lr_backbone * decay ** (top - d), []).append(p)
        seen.add(id(p))


def param_groups(model: SpecialistModel, tc: Any) -> list[dict]:
    enc = pretrained_encoders(model.backbone)
    groups: dict[float, list] = {}
    seen: set[int] = set()
    for m in enc.values():
        _encoder_groups(m, tc, groups, seen)
    scratch = tc.lr_head if enc else tc.lr_backbone
    for p in model.backbone.parameters():
        if id(p) not in seen:
            groups.setdefault(scratch, []).append(p)
    for h in model.heads.values():
        groups.setdefault(tc.lr_head, []).extend(h.parameters())
    return [{"params": ps, "lr": lr} for lr, ps in groups.items()]


def amp_dtype(spec: TaskSpec, model: torch.nn.Module, dev: str) -> torch.dtype | None:
    p = spec.train.precision
    if p == "auto":
        if dev not in ("mps", "cuda") or not pretrained_encoders(model):
            return None
        return torch.bfloat16 if dev == "cuda" and torch.cuda.is_bf16_supported() else torch.float16
    return {"fp32": None, "bf16": torch.bfloat16, "fp16": torch.float16}[p]


def autocast(dev: str, dtype: torch.dtype | None) -> contextlib.AbstractContextManager:
    return torch.autocast(dev, dtype=dtype) if dtype is not None else contextlib.nullcontext()
