from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .infosets import PAD

SLOTS = 11
SEPARATOR = 15
PAD_TOKEN = 16
START = 17
VOCAB = 18
STREETS = 4
EMBED = 64
HIDDEN = 256
BATCH = 8192
EPOCHS = 4
LEARNING_RATE = 2e-3
HELD_SHARE = 0.02
CHUNK = 1 << 18
Log = Callable[[str], None]


class Net(nn.Module):
    def __init__(self, span: int):
        super().__init__()
        self.span = span
        self.tokens = nn.Embedding(VOCAB, EMBED, padding_idx=PAD_TOKEN)
        self.cards = nn.Embedding(STREETS * span, EMBED)
        self.reader = nn.GRU(EMBED, HIDDEN, batch_first=True)
        self.head = nn.Sequential(nn.Linear(HIDDEN + EMBED, HIDDEN), nn.ReLU(), nn.Linear(HIDDEN, SLOTS))

    def forward(
        self, tokens: torch.Tensor, last: torch.Tensor, cards: torch.Tensor, legal: torch.Tensor
    ) -> torch.Tensor:
        read, _ = self.reader(self.tokens(tokens))
        state = read[torch.arange(len(tokens), device=tokens.device), last]
        logits = self.head(torch.cat([state, self.cards(cards)], dim=-1))
        return logits.masked_fill(~legal, float("-inf"))


def slots_of(rows: np.ndarray) -> np.ndarray:
    offered = rows["offered"].astype(np.int64)
    return np.where(offered == PAD, SLOTS, offered)


def inputs(rows: np.ndarray, span: int, device: str) -> dict[str, torch.Tensor]:
    hist = rows["hist"].astype(np.int64)
    street = (hist == SEPARATOR).sum(1)
    hist[hist == PAD] = PAD_TOKEN
    tokens = np.concatenate([np.full((len(rows), 1), START), hist], axis=1)
    legal = np.zeros((len(rows), SLOTS + 1), bool)
    np.put_along_axis(legal, slots_of(rows), True, axis=1)
    found = {
        "tokens": tokens,
        "last": rows["tokens"].astype(np.int64),
        "cards": street * span + rows["bucket"].astype(np.int64),
        "legal": legal[:, :SLOTS],
    }
    return {k: torch.from_numpy(v).to(device) for k, v in found.items()}


def targets(rows: np.ndarray, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    spread = np.zeros((len(rows), SLOTS + 1), np.float32)
    np.put_along_axis(spread, slots_of(rows), rows["probs"], axis=1)
    spread = spread[:, :SLOTS]
    spread /= np.maximum(spread.sum(1, keepdims=True), 1e-9)
    return torch.from_numpy(spread).to(device), torch.from_numpy(rows["visits"].astype(np.float32)).to(device)


def scored(net: Net, rows: np.ndarray, device: str) -> dict[str, float]:
    with torch.no_grad():
        logp = torch.log_softmax(net(**inputs(rows, net.span, device)), dim=-1)
        target, weight = targets(rows, device)
        gap = torch.where(target > 0, target * (torch.log(target.clamp_min(1e-9)) - logp), torch.zeros_like(target))
        same = (logp.argmax(-1) == target.argmax(-1)).float()
        total = weight.sum()
        return {"kl": float((gap.sum(-1) * weight).sum() / total), "top_move": float((same * weight).sum() / total)}


def fit(records: np.ndarray, span: int, log: Log, device: str = "cpu", epochs: int = EPOCHS, seed: int = 0) -> tuple:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    seen = np.array(records[np.flatnonzero(records["visits"] > 0)])
    rng.shuffle(seen)
    held = max(1, int(len(seen) * HELD_SHARE))
    test, train = seen[:held], seen[held:]
    net = Net(span).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=LEARNING_RATE)
    for epoch in range(epochs):
        net.train()
        rng.shuffle(train)
        for i in range(0, len(train), BATCH):
            rows = train[i : i + BATCH]
            logp = torch.log_softmax(net(**inputs(rows, span, device)), dim=-1)
            target, weight = targets(rows, device)
            loss = -((target * logp.masked_fill(target == 0, 0.0)).sum(-1) * weight).sum() / weight.sum()
            opt.zero_grad()
            loss.backward()
            opt.step()
        net.eval()
        if epoch == epochs - 1 or (epoch + 1) % max(1, epochs // 4) == 0:
            at = scored(net, test, device)
            log(f"  epoch {epoch + 1}/{epochs}: held-out KL {at['kl']:.4f}, same top move {at['top_move']:.1%}")
    return net, {**scored(net, test, device), "trained_on": len(train), "held_out": len(test)}


def table(net: Net, records: np.ndarray, path: Path, device: str) -> None:
    with path.open("wb") as out, torch.no_grad():
        for i in range(0, len(records), CHUNK):
            rows = np.array(records[i : i + CHUNK])
            probs = torch.softmax(net(**inputs(rows, net.span, device)), dim=-1)
            padded = torch.cat([probs, torch.zeros(len(rows), 1, device=device)], dim=1)
            picked = padded.gather(1, torch.from_numpy(slots_of(rows)).to(device))
            out.write(picked.cpu().numpy().astype("<f4").tobytes())
