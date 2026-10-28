from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch

from . import checkpoint
from .patch import BATCH, LR, SEED, _device, reporter, weights_path
from .patch_net import MAX_HORIZON, SIZES, PatchNet, Pick, Size, picker, train
from .recipe import V1, Recipe
from .synthetic import Synthetic

CORPUS_SHARE = 0.8


class Corpus:
    def __init__(self, paths: list[str], length: int, patch: int) -> None:
        from datasets import Dataset

        self.columns = [Dataset.from_file(p).data.table.column("target") for p in paths]
        self.length, self.patch = length, patch

    def draw(self, rng: np.random.Generator) -> np.ndarray | None:
        import pyarrow as pa

        column = self.columns[int(rng.integers(len(self.columns)))]
        values = column[int(rng.integers(len(column)))].values
        if pa.types.is_list(values.type) or pa.types.is_large_list(values.type):
            values = values[int(rng.integers(len(values)))].values
        series = values.to_numpy(zero_copy_only=False).astype(float)
        if len(series) <= MAX_HORIZON + self.patch or np.isnan(series).all():
            return None
        start = int(rng.integers(0, max(1, len(series) - self.length)))
        return series[start : start + self.length]


def mixed(source: Corpus | Synthetic, own: Pick, share: float) -> Pick:
    def pick(rng: np.random.Generator, n: int) -> list[tuple[np.ndarray, int]]:
        drawn: list[tuple[np.ndarray, int]] = []
        while len(drawn) < rng.binomial(n, share):
            segment = source.draw(rng)
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


def _sources(size: Size, paths: list[str], recipe: Recipe, log: Callable[[str], None]) -> Pick:
    own, length = picker(_training_portions(log), size.patch), size.context + MAX_HORIZON
    pick = mixed(Corpus(paths, length, size.patch), own, CORPUS_SHARE) if paths else own
    log(f"  GIFT-Eval training portions plus {len(paths)} pretraining corpus files")
    if not recipe.synthetic:
        return pick
    log(f"  {recipe.synthetic:.0%} of every batch drawn from synthetic series")
    return mixed(Synthetic(length), pick, recipe.synthetic)


def pretrain(log: Callable[[str], None], steps: int, size_name: str = "base", recipe: Recipe = V1) -> Path:
    from ..train.gpu import gpu_slot
    from .corpus import files

    size, paths, name = SIZES[size_name], files(), recipe.weights(size_name)
    pick = _sources(size, paths, recipe, log)
    torch.manual_seed(SEED)
    net, device = PatchNet(size, len(recipe.quantiles)), _device()
    millions = sum(p.numel() for p in net.parameters()) / 1e6
    log(f"  recipe {recipe.name}: {millions:.1f}M parameters, {steps} steps on {device}")

    with gpu_slot(device, log):
        resume = checkpoint.path(checkpoint.key("pretraining", name, steps, sorted(paths)))
        train(net, pick, steps, LR, BATCH, device, SEED, reporter("pretraining", steps, log), resume, recipe)
    path = weights_path(name)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(net.state_dict(), path)
    log(f"  saved {path}")
    return path
