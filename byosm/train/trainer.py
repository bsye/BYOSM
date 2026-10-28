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
from ..text import prefix
from .common import amp_dtype, autocast, batches, param_groups, pretrained_encoders, to_device
from .regularise import mixed_logits, symmetric_kl
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
    hooks: dict[str, Callable[..., Any]] = field(default_factory=dict)
    frozen: list[torch.nn.Module] = field(init=False)
    probe: int = field(init=False)
    steps: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        tc = self.spec.train
        enc = pretrained_encoders(self.model.backbone)
        self.frozen = list(enc.values()) if enc else [self.model.backbone]
        self.opt = torch.optim.AdamW(param_groups(self.model, tc), weight_decay=0.01)
        steps_per_epoch = math.ceil(len(self.train_raw) / tc.batch_size)
        self.total = tc.epochs * steps_per_epoch
        self.sched = warmup_cosine(self.opt, self.total, self.total // 15)
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

    def _view(self, raw: Any) -> Any:
        rate = self.spec.train.prefix_rate
        cut = rate > 0 and self.spec.modality == "text" and self.rng.random() < rate
        return prefix(raw, self.rng) if cut else raw

    def _loss(self, b: list[int]) -> torch.Tensor:
        tc, dev = self.spec.train, self.dev
        fl = [self.train_t[i] for i in b] if self.spec.task_type == "frame" else None
        inputs, ft = self.mod.collate([self._view(self.train_raw[i]) for i in b], fl)
        with autocast(dev, self.amp):
            logits, mask = self.model(to_device(inputs, dev))
        logits = logits.float()
        y = ft.to(dev) if ft is not None else torch.tensor([self.train_t[i] for i in b], device=dev)
        has_gold = bool((y != IGNORE).any())
        alpha = tc.distill_alpha if self.soft is not None else 1.0
        loss = self._task_loss(logits, y, mask) * alpha if has_gold else logits.sum() * 0
        pooled_task = self.spec.task_type == "classify" and self.soft is None and has_gold
        if pooled_task and tc.rdrop > 0:
            loss = loss + tc.rdrop * self._rdrop(inputs, logits)
        if pooled_task and tc.mixup > 0:
            loss = loss + self._mixup(inputs, y)
        if self.soft is not None:
            tp, valid = soft_batch(self.spec, self.soft, b, logits, mask, dev)
            loss = loss + (1 - tc.distill_alpha) * distill_loss(
                logits.float(), tp.float(), valid, tc.distill_temperature
            )
        return loss

    def _task_loss(self, logits: torch.Tensor, y: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        custom = self.hooks.get("loss") if self.spec.task_type == "classify" else None
        return custom(logits, y, self.weight, self.steps) if custom else task_loss(logits, y, mask, self.weight)

    def _rdrop(self, inputs: dict, logits: torch.Tensor) -> torch.Tensor:
        with autocast(self.dev, self.amp):
            again, _ = self.model(to_device(inputs, self.dev))
        return symmetric_kl(logits, again.float())

    def _mixup(self, inputs: dict, y: torch.Tensor) -> torch.Tensor:
        with autocast(self.dev, self.amp):
            mixed, perm, lam = mixed_logits(self.model, to_device(inputs, self.dev), self.spec.train.mixup, self.rng)
        mixed = mixed.float()
        none = torch.ones(1, dtype=torch.bool, device=self.dev)
        return lam * task_loss(mixed, y, none, self.weight) + (1 - lam) * task_loss(mixed, y[perm], none, self.weight)

    def _step(self, loss: torch.Tensor) -> None:
        self.opt.zero_grad(set_to_none=True)
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.opt)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.scaler.step(self.opt)
        self.scaler.update()
        self.sched.step()
        if "after_step" in self.hooks:
            self.hooks["after_step"](self.model, self.steps, self.total)
        self.steps += 1
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
            losses.append(loss.item())
        return losses
