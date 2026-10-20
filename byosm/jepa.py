from __future__ import annotations

import random
from collections.abc import Callable
from typing import Any

import torch
from rich.progress import Progress

from .model import SpecialistModel, warmup_cosine


def sigreg(z: torch.Tensor, num_slices: int = 256, knots: int = 17, t_max: float = 3.0) -> torch.Tensor:
    n, k = z.shape
    a = torch.randn(k, num_slices, device=z.device, dtype=z.dtype)
    a = a / a.norm(dim=0, keepdim=True)
    s = z @ a
    t = torch.linspace(-t_max, t_max, knots, device=z.device, dtype=z.dtype)
    x = s.unsqueeze(-1) * t
    gauss = torch.exp(-0.5 * t**2)
    err = (torch.cos(x).mean(0) - gauss) ** 2 + torch.sin(x).mean(0) ** 2
    stat = torch.trapezoid(err * gauss, t, dim=-1) * n
    return stat.mean()


def lejepa_loss(
    model: SpecialistModel, view_batches: list[dict[str, torch.Tensor]], n_global: int, lam: float
) -> tuple[torch.Tensor, dict[str, float]]:
    z = torch.stack([model.embed(vb) for vb in view_batches])
    centers = z[:n_global].mean(0)
    pred = (z - centers).pow(2).mean()
    sig = torch.stack([sigreg(zv) for zv in z]).mean()
    loss = (1 - lam) * pred + lam * sig
    return loss, {
        "pred": float(pred.detach()),
        "sigreg": float(sig.detach()),
        "emb_std": float(z.detach().std(1).mean()),
    }


def _view_batches(
    modality: Any, raws: list[Any], rng: random.Random, device: str
) -> tuple[list[dict[str, torch.Tensor]], int]:
    per = [modality.views(r, rng) for r in raws]
    n_global, n_local = len(per[0][0]), len(per[0][1])
    view_batches = []
    for v in range(n_global + n_local):
        xs = [(glob + loc)[v] for glob, loc in per]
        inputs, _ = modality.collate(xs)
        view_batches.append({k: t.to(device) for k, t in inputs.items()})
    return view_batches, n_global


def _log_step(log: Callable[[str], None], step: int, steps: int, loss: float, parts: dict[str, float]) -> None:
    log(
        f"  jepa step {step}/{steps} loss={loss:.4f} pred={parts['pred']:.4f} "
        f"sigreg={parts['sigreg']:.4f} emb_std={parts['emb_std']:.3f}"
    )


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
    sched = warmup_cosine(opt, steps, steps // 10)
    history = []
    bs = min(batch_size, len(unlabeled))
    with Progress(transient=True) as prog:
        task = prog.add_task("LeJEPA pretraining", total=steps)
        for step in range(steps):
            view_batches, n_global = _view_batches(modality, rng.sample(unlabeled, bs), rng, device)
            loss, parts = lejepa_loss(model, view_batches, n_global, lam)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            history.append({"step": step, "loss": float(loss), **parts})
            prog.update(task, advance=1, description=f"LeJEPA pretraining loss={float(loss):.3f}")
            if step % max(1, steps // 5) == 0:
                _log_step(log, step, steps, float(loss), parts)
    return history
