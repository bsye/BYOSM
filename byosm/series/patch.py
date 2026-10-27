from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch

from . import checkpoint
from .models import Group
from .patch_net import BASE, SIZES, SMALL, PatchNet, Size, picker, train, window

STEPS = 3000
FINETUNE_STEPS = 800
BATCH = 256
LR = 1e-3
FINETUNE_LR = 2e-4
SEED = 0


def weights_path(name: str = "base") -> Path:
    from ..config import cache_root

    return cache_root() / "series" / f"patch-{name}.pt"


def pretrained() -> tuple[str, Size] | None:
    return next(((n, s) for n, s in reversed(SIZES.items()) if weights_path(n).exists()), None)


def missing_size() -> str | None:
    return next((n for n in SIZES if not weights_path(n).exists()), None)


def reporter(label: str, steps: int, log: Callable[[str], None] = print) -> Callable[[int, float], None]:
    from ..events import emit

    def report(step: int, loss: float) -> None:
        log(f"  {label}: step {step}/{steps} loss {loss:.4f}")
        emit("training", step=step, steps=steps, loss=round(loss, 4))

    return report


def _device() -> str:
    from ..config import device

    return device()


class PatchModel:
    name = "patch transformer"
    size: Size = SMALL
    pretrained = False

    def __init__(self, effort: int = 0) -> None:
        self.effort = effort

    def fit(self, groups: list[Group]) -> None:
        from ..train.gpu import gpu_slot

        torch.manual_seed(SEED)
        self.net = PatchNet(self.size)
        steps, lr = STEPS, LR
        if self.pretrained:
            self.net.load_state_dict(torch.load(weights_path(self.weights), map_location="cpu"))
            steps, lr = FINETUNE_STEPS, FINETUNE_LR
        with gpu_slot(_device(), print, fallback="cpu" if self.size == SMALL else None) as device:
            series, total = [(h, g.horizon) for g in groups for h in g.histories], steps * 2**self.effort
            report = reporter(self.name, total)
            resume = checkpoint.path(checkpoint.key(self.name, self.size, total, lr, arrays=[h for h, _ in series]))
            train(self.net, picker(series, self.size.patch), total, lr, BATCH, device, SEED, report, resume)

    def predict_many(self, histories: list[np.ndarray], season: int, horizon: int) -> list[np.ndarray]:
        windows = [window(h, self.size.context) for h in histories]
        out = []
        with torch.no_grad():
            for start in range(0, len(windows), BATCH):
                part = windows[start : start + BATCH]
                v = torch.tensor(np.asarray([w[0] for w in part]), dtype=torch.float32)
                m = torch.tensor(np.asarray([w[1] for w in part]), dtype=torch.float32)
                pred = self.net(v, m).numpy()[:, :, :horizon]
                out += [np.sort(p * w[3] + w[2], axis=0) for p, w in zip(pred, part, strict=True)]
        return out


class PretrainedPatch(PatchModel):
    name = "patch transformer, pretrained"
    pretrained = True

    def __init__(self, effort: int = 0) -> None:
        super().__init__(effort)
        self.weights, self.size = pretrained() or ("base", BASE)
