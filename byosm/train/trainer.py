from __future__ import annotations

import math
import random
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from ..modalities import IGNORE, Modality
from ..model import SpecialistModel, distill_loss, task_loss, warmup_cosine
from ..spec import TaskSpec
from .common import amp_dtype, autocast, batches, param_groups, pretrained_encoders, to_device
from .teacher import soft_batch


def class_weight(spec: TaskSpec, gold: list[Any], dev: str) -> torch.Tensor:
    flat = np.concatenate([np.asarray(t).ravel() for t in gold]) if spec.task_type == "frame" else np.array(gold)
    counts = np.bincount(flat, minlength=len(spec.labels)).astype(float)
    return torch.tensor((counts.sum() / np.maximum(counts, 1) / len(counts)) ** 0.5, dtype=torch.float32, device=dev)


@dataclass
class Trainer:
    spec: TaskSpec
    mod: Modality
    model: SpecialistModel
    dev: str
    train_raw: list[Any]
    train_t: list[Any]
    soft: list[Any] | None
    weight: torch.Tensor
    rng: random.Random
    frozen: list[torch.nn.Module] = field(init=False)
    probe: int = field(init=False)

    def __post_init__(self) -> None:
        tc = self.spec.train
        enc = pretrained_encoders(self.model.backbone)
        self.frozen = list(enc.values()) if enc else [self.model.backbone]
        self.opt = torch.optim.AdamW(param_groups(self.model, tc), weight_decay=0.01)
        steps_per_epoch = math.ceil(len(self.train_raw) / tc.batch_size)
        steps = tc.epochs * steps_per_epoch
        self.sched = warmup_cosine(self.opt, steps, steps // 15)
        self.amp = amp_dtype(self.spec, self.model, self.dev)
        self.scaler = torch.amp.GradScaler(self.dev, enabled=self.amp == torch.float16)
        self.probe = 0 if tc.freeze_backbone or not enc else min(tc.probe_epochs, tc.epochs // 4)
        self.ema = self._ema(steps_per_epoch) if tc.ema else None

    def _ema(self, steps_per_epoch: int) -> torch.optim.swa_utils.AveragedModel:
        avg = torch.optim.swa_utils.get_ema_multi_avg_fn(1 - 1 / max(10, steps_per_epoch / 2))
        return torch.optim.swa_utils.AveragedModel(self.model, multi_avg_fn=avg, use_buffers=True)

    def log_recipe(self, log: Callable[[str], None]) -> None:
        if self.probe or self.amp or self.ema:
            precision = str(self.amp).replace("torch.", "") if self.amp else "fp32"
            log(
                f"  recipe: {self.probe} probe epoch(s), layer LR floor {self.spec.train.layer_decay_floor}, "
                f"ema={'on' if self.ema else 'off'}, precision={precision}"
            )

    @property
    def selected(self) -> torch.nn.Module:
        return self.ema.module if self.ema is not None else self.model

    def _freeze(self, on: bool) -> None:
        for m in self.frozen:
            for p in m.parameters():
                p.requires_grad = not on

    def _loss(self, b: list[int]) -> torch.Tensor:
        tc, dev = self.spec.train, self.dev
        fl = [self.train_t[i] for i in b] if self.spec.task_type == "frame" else None
        inputs, ft = self.mod.collate([self.train_raw[i] for i in b], fl)
        with autocast(dev, self.amp):
            logits, mask = self.model(to_device(inputs, dev))
        logits = logits.float()
        y = ft.to(dev) if ft is not None else torch.tensor([self.train_t[i] for i in b], device=dev)
        has_gold = bool((y != IGNORE).any())
        alpha = tc.distill_alpha if self.soft is not None else 1.0
        loss = task_loss(logits, y, mask, self.weight) * alpha if has_gold else logits.sum() * 0
        if self.soft is not None:
            tp, valid = soft_batch(self.spec, self.soft, b, logits, mask, dev)
            loss = loss + (1 - tc.distill_alpha) * distill_loss(
                logits.float(), tp.float(), valid, tc.distill_temperature
            )
        return loss

    def _step(self, loss: torch.Tensor) -> None:
        self.opt.zero_grad(set_to_none=True)
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.opt)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.scaler.step(self.opt)
        self.scaler.update()
        self.sched.step()
        if self.ema is not None:
            self.ema.update_parameters(self.model)

    def epoch(self, ep: int) -> list[float]:
        tc = self.spec.train
        frz = tc.freeze_backbone or ep < self.probe
        self._freeze(frz)
        self.model.train()
        for m in self.frozen if frz else []:
            m.eval()
        losses = []
        for b in batches(len(self.train_raw), tc.batch_size, True, self.rng):
            loss = self._loss(b)
            self._step(loss)
            losses.append(float(loss))
        return losses
