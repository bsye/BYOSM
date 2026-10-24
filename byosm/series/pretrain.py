from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch

from .patch import BATCH, LR, SEED, _device, weights_path
from .patch_net import BASE, MAX_HORIZON, PatchNet, Pick, picker, train

CORPUS_SHARE = 0.8


class Corpus:
    def __init__(self, paths: list[str], length: int) -> None:
        from datasets import Dataset

        self.columns = [Dataset.from_file(p).data.table.column("target") for p in paths]
        self.length = length

    def draw(self, rng: np.random.Generator) -> np.ndarray | None:
        import pyarrow as pa

        column = self.columns[int(rng.integers(len(self.columns)))]
        values = column[int(rng.integers(len(column)))].values
        if pa.types.is_list(values.type) or pa.types.is_large_list(values.type):
            values = values[int(rng.integers(len(values)))].values
        series = values.to_numpy(zero_copy_only=False).astype(float)
        if len(series) <= MAX_HORIZON + BASE.patch or np.isnan(series).all():
            return None
        start = int(rng.integers(0, max(1, len(series) - self.length)))
        return series[start : start + self.length]


def mixed(corpus: Corpus, own: Pick) -> Pick:
    def pick(rng: np.random.Generator, n: int) -> list[tuple[np.ndarray, int]]:
        drawn: list[tuple[np.ndarray, int]] = []
        while len(drawn) < rng.binomial(n, CORPUS_SHARE):
            segment = corpus.draw(rng)
            if segment is not None:
                drawn.append((segment, MAX_HORIZON))
        return drawn + own(rng, n - len(drawn))

    return pick


def _training_portions(log: Callable[[str], None]) -> list[tuple[np.ndarray, int]]:
    from .gifteval import all_configs, horizon, load, windows

    out = []
    for config in all_configs():
        try:
            series = load(config)
        except Exception as e:
            log(f"  skip {config}: {type(e).__name__}")
            continue
        cut = 2 * windows(series) * horizon(series[0].freq)
        out += [(s.values[: len(s.values) - cut], MAX_HORIZON) for s in series if len(s.values) > cut]
    return out


def pretrain(log: Callable[[str], None], steps: int) -> Path:
    from ..events import emit
    from ..train.gpu import gpu_slot
    from .corpus import files

    own = picker(_training_portions(log), BASE.patch)
    paths = files()
    pick = mixed(Corpus(paths, BASE.context + MAX_HORIZON), own) if paths else own
    log(f"  GIFT-Eval training portions plus {len(paths)} pretraining corpus files")
    torch.manual_seed(SEED)
    net, device = PatchNet(BASE), _device()
    log(f"  {sum(p.numel() for p in net.parameters()) / 1e6:.1f}M parameters, {steps} steps on {device}")

    def report(step: int, loss: float) -> None:
        log(f"  step {step}/{steps} loss {loss:.4f}")
        emit("training", step=step, steps=steps, loss=round(loss, 4))

    with gpu_slot(device, log):
        train(net, pick, steps, LR, BATCH, device, SEED, report)
    path = weights_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(net.state_dict(), path)
    log(f"  saved {path}")
    return path
