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
STEPS = 3000
BATCH = 256
LR = 1e-3
SEED = 0
DEVICE = "cpu"


class PatchNet(nn.Module):
    def __init__(self, horizon: int):
        super().__init__()
        self.horizon, patches = horizon, CONTEXT // PATCH
        self.embed = nn.Linear(2 * PATCH, WIDTH)
        self.position = nn.Parameter(torch.zeros(patches, WIDTH))
        layer = nn.TransformerEncoderLayer(WIDTH, HEADS, 4 * WIDTH, dropout=0.1, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, LAYERS)
        self.head = nn.Linear(patches * WIDTH, horizon * len(QUANTILES))

    def forward(self, values: torch.Tensor, seen: torch.Tensor) -> torch.Tensor:
        v, m = values.unfold(1, PATCH, PATCH), seen.unfold(1, PATCH, PATCH)
        x = self.encoder(self.embed(torch.cat([v, m], -1)) + self.position)
        return self.head(x.flatten(1)).view(-1, len(QUANTILES), self.horizon)


def _window(history: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float]:
    ctx = history[-CONTEXT:]
    ctx = np.concatenate([np.full(CONTEXT - len(ctx), np.nan), ctx])
    seen = ~np.isnan(ctx)
    mean = float(np.mean(ctx[seen])) if seen.any() else 0.0
    std = float(np.std(ctx[seen])) if seen.any() else 1.0
    std = std or 1.0
    return np.where(seen, (ctx - mean) / std, 0.0), seen.astype(float), mean, std


def _pinball(pred: torch.Tensor, target: torch.Tensor, known: torch.Tensor) -> torch.Tensor:
    levels = torch.tensor(QUANTILES).view(1, -1, 1)
    diff = target.unsqueeze(1) - pred
    loss = torch.maximum(levels * diff, (levels - 1) * diff) * known.unsqueeze(1)
    return loss.sum() / known.sum().clamp(min=1)


class PatchModel:
    name = "patch transformer"

    def fit(self, histories: list[np.ndarray], season: int, horizon: int) -> None:
        torch.manual_seed(SEED)
        rng = np.random.default_rng(SEED)
        self.horizon = horizon
        usable = [h for h in histories if len(h) > horizon + PATCH]
        self.net = PatchNet(horizon).to(DEVICE)
        opt = torch.optim.AdamW(self.net.parameters(), lr=LR, weight_decay=0.01)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, LR, total_steps=STEPS)
        self.net.train()
        for _ in range(STEPS):
            values, seen, targets, known = [], [], [], []
            for i in rng.integers(0, len(usable), BATCH):
                h = usable[i]
                end = int(rng.integers(PATCH, len(h) - horizon + 1))
                v, m, mean, std = _window(h[:end])
                future = (h[end : end + horizon] - mean) / std
                values.append(v)
                seen.append(m)
                targets.append(np.nan_to_num(future))
                known.append(~np.isnan(future))
            pred = self.net(*(torch.tensor(np.asarray(a), dtype=torch.float32, device=DEVICE) for a in (values, seen)))
            loss = _pinball(
                pred,
                torch.tensor(np.asarray(targets), dtype=torch.float32),
                torch.tensor(np.asarray(known), dtype=torch.float32),
            )
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
            opt.step()
            sched.step()
        self.net.eval()

    def predict(self, history: np.ndarray, horizon: int) -> np.ndarray:
        return self.predict_many([history], horizon)[0]

    def predict_many(self, histories: list[np.ndarray], horizon: int) -> list[np.ndarray]:
        windows = [_window(h) for h in histories]
        out = []
        with torch.no_grad():
            for start in range(0, len(windows), BATCH):
                part = windows[start : start + BATCH]
                v = torch.tensor(np.asarray([w[0] for w in part]), dtype=torch.float32)
                m = torch.tensor(np.asarray([w[1] for w in part]), dtype=torch.float32)
                pred = self.net(v, m).numpy()
                out += [np.sort(p * w[3] + w[2], axis=0) for p, w in zip(pred, part, strict=True)]
        return out
