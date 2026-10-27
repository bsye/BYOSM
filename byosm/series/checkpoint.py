from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import torch

EVERY = 5000
FOLDER = "checkpoints"
SUFFIX = ".pt"


def key(*parts: Any, arrays: list[np.ndarray] = ()) -> str:
    digest = hashlib.sha1(repr(parts).encode())
    for a in arrays:
        digest.update(np.ascontiguousarray(a).tobytes())
    return digest.hexdigest()[:16]


def path(name: str) -> Path:
    from ..config import cache_root

    return cache_root() / "series" / FOLDER / f"{name}{SUFFIX}"


def save(out: Path, step: int, net: torch.nn.Module, opt: Any, sched: Any, rng: np.random.Generator) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "step": step,
        "net": net.state_dict(),
        "opt": opt.state_dict(),
        "sched": sched.state_dict(),
        "rng": rng.bit_generator.state,
        "torch_rng": torch.get_rng_state(),
    }
    torch.save(state, out.with_suffix(".tmp"))
    out.with_suffix(".tmp").replace(out)


def restore(src: Path, net: torch.nn.Module, opt: Any, sched: Any, rng: np.random.Generator, device: str) -> int:
    if not src.exists():
        return 0
    state = torch.load(src, map_location=device, weights_only=False)
    net.load_state_dict(state["net"])
    opt.load_state_dict(state["opt"])
    sched.load_state_dict(state["sched"])
    rng.bit_generator.state = state["rng"]
    torch.set_rng_state(state["torch_rng"].cpu())
    return int(state["step"])
