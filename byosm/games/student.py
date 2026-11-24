from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .decision import Decision
from .game import Policy

HIDDEN = 64
EPOCHS = 600
LEARNING_RATE = 3e-3


@dataclass(frozen=True)
class Student:
    net: nn.Module
    slots: int

    def policy(self, decisions: list[Decision]) -> Policy:
        with torch.no_grad():
            got = torch.softmax(masked(self.net(features(decisions)), decisions, self.slots), dim=-1)
        return {d.key: [float(got[i, a]) for a in d.legal] for i, d in enumerate(decisions)}


def features(decisions: list[Decision]) -> torch.Tensor:
    return torch.tensor([d.features for d in decisions], dtype=torch.float32)


def masked(logits: torch.Tensor, decisions: list[Decision], slots: int) -> torch.Tensor:
    legal = torch.zeros(len(decisions), slots, dtype=torch.bool)
    for i, d in enumerate(decisions):
        legal[i, d.legal] = True
    return logits.masked_fill(~legal, float("-inf"))


def soft_loss(logits: torch.Tensor, target: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    logp = torch.log_softmax(logits, dim=-1).masked_fill(target == 0, 0.0)
    return -((target * logp).sum(-1) * weight).sum() / weight.sum()


def train(decisions: list[Decision], slots: int, seed: int = 0) -> Student:
    torch.manual_seed(seed)
    x = features(decisions)
    target = torch.tensor([d.soft for d in decisions], dtype=torch.float32)
    weight = torch.tensor([d.weight for d in decisions], dtype=torch.float32)
    net = nn.Sequential(
        nn.Linear(x.shape[1], HIDDEN), nn.ReLU(), nn.Linear(HIDDEN, HIDDEN), nn.ReLU(), nn.Linear(HIDDEN, slots)
    )
    opt = torch.optim.Adam(net.parameters(), lr=LEARNING_RATE)
    for _ in range(EPOCHS):
        loss = soft_loss(masked(net(x), decisions, slots), target, weight)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return Student(net.eval(), slots)
