from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch

from .patch_net import MAX_HORIZON, PatchNet, train, window

STEPS = 3000
FINETUNE_STEPS = 800
PRETRAIN_STEPS = 20000
BATCH = 256
LR = 1e-3
FINETUNE_LR = 2e-4
SEED = 0
WEIGHTS = "patch-pretrained.pt"


def weights_path() -> Path:
    from ..config import cache_root

    return cache_root() / "series" / WEIGHTS


def _device() -> str:
    from ..config import device

    return device()


class PatchModel:
    name = "patch transformer"
    pretrained = False

    def fit(self, histories: list[np.ndarray], season: int, horizon: int) -> None:
        from ..train.gpu import gpu_slot

        torch.manual_seed(SEED)
        self.net = PatchNet()
        steps, lr = STEPS, LR
        if self.pretrained:
            self.net.load_state_dict(torch.load(weights_path(), map_location="cpu"))
            steps, lr = FINETUNE_STEPS, FINETUNE_LR
        device = _device()
        with gpu_slot(device, lambda _: None):
            train(self.net, histories, horizon, steps, lr, BATCH, device, SEED)

    def predict(self, history: np.ndarray, horizon: int) -> np.ndarray:
        return self.predict_many([history], horizon)[0]

    def predict_many(self, histories: list[np.ndarray], horizon: int) -> list[np.ndarray]:
        windows = [window(h) for h in histories]
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


def pretrain(log: Callable[[str], None]) -> Path:
    from .gifteval import all_configs, horizon, load, windows

    histories = []
    for config in all_configs():
        try:
            series = load(config)
        except Exception as e:
            log(f"  skip {config}: {type(e).__name__}")
            continue
        cut = windows(series) * horizon(series[0].freq)
        histories += [s.values[: len(s.values) - cut] for s in series if len(s.values) > cut]
        log(f"  {config}: {len(series)} series, training portion only")
    torch.manual_seed(SEED)
    net, device = PatchNet(), _device()
    from ..train.gpu import gpu_slot

    log(f"  pretraining on {len(histories)} series for {PRETRAIN_STEPS} steps on {device}")
    with gpu_slot(device, log):
        train(net, histories, MAX_HORIZON, PRETRAIN_STEPS, LR, BATCH, device, SEED)
    path = weights_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(net.state_dict(), path)
    log(f"  saved {path}")
    return path
