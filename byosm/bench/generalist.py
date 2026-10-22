from __future__ import annotations

import random
from collections.abc import Callable
from typing import Any

import numpy as np
import torch

from .. import jepa
from ..config import device as pick_device
from ..data.store import Dataset
from ..modalities import get_modality
from ..model import SpecialistModel, task_loss, warmup_cosine
from ..project import Project
from ..spec import TaskSpec
from ..train import batches, cpu_state, evaluate_rows, targets_of, to_device


def _generalist_spec(specs: list[TaskSpec], backbone: str) -> TaskSpec:
    gspec = specs[0].model_copy(deep=True)
    gspec.name, gspec.backbone = "generalist", backbone
    gspec.train.epochs = max(s.train.epochs for s in specs)
    gspec.train.pretrain_steps = max(s.train.pretrain_steps for s in specs)
    return gspec


def _load_tasks(specs: list[TaskSpec]) -> dict[str, dict[str, Any]]:
    tasks = {}
    for s in specs:
        dd = Project(s.name).data_dir_for(s)
        mod = get_modality(s, str(dd))
        ds = Dataset(dd)
        tr, dv = ds.load("train"), ds.load("dev")
        tasks[s.name] = {
            "spec": s,
            "mod": mod,
            "train": [mod.load(r) for r in tr],
            "t": targets_of(s, mod, tr),
            "dev_rows": dv,
            "dev": [mod.load(r) for r in dv],
        }
    return tasks


def _plan(tasks: dict[str, dict[str, Any]], batch_size: int, rng: random.Random) -> list[tuple[str, list[int]]]:
    return [(n, b) for n, t in tasks.items() for b in batches(len(t["train"]), batch_size, True, rng)]


def _optimizer(model: SpecialistModel, tc: Any, steps: int) -> tuple[torch.optim.Optimizer, Any]:
    head_params = [p for h in model.heads.values() for p in h.parameters()]
    opt = torch.optim.AdamW(
        [
            {"params": head_params, "lr": tc.lr_head},
            {"params": list(model.backbone.parameters()), "lr": tc.lr_backbone},
        ],
        weight_decay=0.01,
    )
    return opt, warmup_cosine(opt, steps, steps // 15)


def _train_step(
    model: SpecialistModel, gmod: Any, t: dict[str, Any], name: str, b: list[int], dev: str
) -> torch.Tensor:
    fl = [t["t"][i] for i in b] if t["spec"].task_type == "frame" else None
    inputs, ft = gmod.collate([t["train"][i] for i in b], fl)
    logits, mask = model(to_device(inputs, dev), name)
    y = ft.to(dev) if ft is not None else torch.tensor([t["t"][i] for i in b], device=dev)
    return task_loss(logits, y, mask)


def _train_epoch(
    model: SpecialistModel,
    gmod: Any,
    tasks: dict[str, dict[str, Any]],
    batch_size: int,
    rng: random.Random,
    opt: torch.optim.Optimizer,
    sched: Any,
    dev: str,
) -> None:
    model.train()
    plan = _plan(tasks, batch_size, rng)
    rng.shuffle(plan)
    for name, b in plan:
        loss = _train_step(model, gmod, tasks[name], name, b, dev)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()


def _dev_scores(
    model: SpecialistModel, gmod: Any, tasks: dict[str, dict[str, Any]], dev: str
) -> tuple[dict[str, float], dict[str, Any]]:
    objs, thr = {}, {}
    for name, t in tasks.items():
        ev = evaluate_rows(model, gmod, t["spec"], t["dev_rows"], t["dev"], dev, None, calibrate=True, head=name)
        objs[name], thr[name] = ev["metrics"]["objective"], ev["threshold"]
    return objs, thr


def train_generalist(
    specs: list[TaskSpec], backbone: str, log: Callable[[str], None] = print, seed: int = 0
) -> tuple[SpecialistModel, dict[str, Any], TaskSpec]:
    torch.manual_seed(seed)
    rng = random.Random(seed)
    dev = pick_device()
    gspec = _generalist_spec(specs, backbone)
    gmod = get_modality(gspec)
    model = SpecialistModel(gmod.build_backbone(), {s.name: (len(s.labels), s.task_type) for s in specs})
    tasks = _load_tasks(specs)
    n_train = sum(len(t["train"]) for t in tasks.values())
    log(f"  generalist: {backbone} with {len(specs)} heads, {n_train} train examples on {dev}")
    tc = gspec.train
    if tc.pretrain_steps > 0:
        unl = [x for t in tasks.values() for x in t["train"]]
        jepa.pretrain(
            model, gmod, unl, tc.pretrain_steps, tc.batch_size, tc.lr_backbone, tc.pretrain_lambda, dev, log, seed
        )
    model.to(dev)
    opt, sched = _optimizer(model, tc, tc.epochs * len(_plan(tasks, tc.batch_size, rng)))
    best, best_score, thresholds = None, -1.0, {}
    for ep in range(tc.epochs):
        _train_epoch(model, gmod, tasks, tc.batch_size, rng, opt, sched, dev)
        objs, thr = _dev_scores(model, gmod, tasks, dev)
        score = float(np.mean(list(objs.values())))
        per_task = " ".join(f"{k}={v:.3f}" for k, v in objs.items())
        log(f"  generalist epoch {ep + 1}/{tc.epochs} mean dev objective={score:.4f} " + per_task)
        if score > best_score:
            best_score, thresholds = score, thr
            best = cpu_state(model)
    model.load_state_dict(best)
    return model, thresholds, gspec
