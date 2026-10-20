"""LeJEPA domain adaptation (Balestriero & LeCun, 2025): multi-view prediction + SIGReg.

Every view's embedding is pulled toward the mean embedding of the global views (prediction term), while SIGReg pushes
the batch's embedding distribution toward an isotropic Gaussian along random 1-D projections (Epps-Pulley test on the
characteristic function). No EMA teacher, no stop-gradient, one hyperparameter (lambda). Modality-agnostic: each
modality plugin only provides `views()`.
"""
from __future__ import annotations

import math
import random
from typing import Any, Callable

import torch
from rich.progress import Progress

from .model import SpecialistModel


def sigreg(z: torch.Tensor, num_slices: int = 256, knots: int = 17, t_max: float = 3.0) -> torch.Tensor:
    """Sketched isotropic-Gaussian regularizer. z: [N, K] embeddings."""
    n, k = z.shape
    a = torch.randn(k, num_slices, device=z.device, dtype=z.dtype)
    a = a / a.norm(dim=0, keepdim=True)
    s = z @ a  # [N, M]
    t = torch.linspace(-t_max, t_max, knots, device=z.device, dtype=z.dtype)
    x = s.unsqueeze(-1) * t  # [N, M, K]
    gauss = torch.exp(-0.5 * t**2)
    err = (torch.cos(x).mean(0) - gauss) ** 2 + torch.sin(x).mean(0) ** 2  # |ECF - CF_N(0,1)|^2
    stat = torch.trapezoid(err * gauss, t, dim=-1) * n
    return stat.mean()


def lejepa_loss(model: SpecialistModel, view_batches: list[dict[str, torch.Tensor]], n_global: int, lam: float) -> tuple[torch.Tensor, dict[str, float]]:
    z = torch.stack([model.embed(vb) for vb in view_batches])  # [V, B, K]
    centers = z[:n_global].mean(0)
    pred = (z - centers).pow(2).mean()  # mean over dims: summing makes collapse cheaper than SIGReg's penalty
    sig = torch.stack([sigreg(zv) for zv in z]).mean()
    loss = (1 - lam) * pred + lam * sig
    # Collapse monitor: per-dim std of embeddings across the batch (SIGReg drives it toward 1).
    return loss, {"pred": float(pred.detach()), "sigreg": float(sig.detach()), "emb_std": float(z.detach().std(1).mean())}


def pretrain(
    model: SpecialistModel,
    modality: Any,
    unlabeled: list[Any],
    steps: int,
    batch_size: int,
    lr: float,
    lam: float,
    device: str,
    log: Callable[[str], None] = print,
    seed: int = 0,
) -> list[dict[str, float]]:
    if steps <= 0 or len(unlabeled) < 8:
        return []
    rng = random.Random(seed)
    model.to(device).train()
    params = list(model.backbone.parameters()) + list(model.proj.parameters())
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / max(1, steps // 10)) * 0.5 * (1 + math.cos(math.pi * s / steps)))
    history = []
    bs = min(batch_size, len(unlabeled))
    with Progress(transient=True) as prog:
        task = prog.add_task("LeJEPA pretraining", total=steps)
        for step in range(steps):
            raws = rng.sample(unlabeled, bs)
            per = [modality.views(r, rng) for r in raws]
            n_global, n_local = len(per[0][0]), len(per[0][1])
            view_batches = []
            for v in range(n_global + n_local):
                xs = [(g + l)[v] for g, l in per]
                inputs, _ = modality.collate(xs)
                view_batches.append({k: t.to(device) for k, t in inputs.items()})
            loss, parts = lejepa_loss(model, view_batches, n_global, lam)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            history.append({"step": step, "loss": float(loss), **parts})
            prog.update(task, advance=1, description=f"LeJEPA pretraining loss={float(loss):.3f}")
            if step % max(1, steps // 5) == 0:
                log(f"  jepa step {step}/{steps} loss={float(loss):.4f} pred={parts['pred']:.4f} sigreg={parts['sigreg']:.4f} emb_std={parts['emb_std']:.3f}")
    return history
