"""The thesis experiment: N tiny specialists vs one shared-backbone generalist trained on the union of the same data.

Same data, same recipe, same held-out test sets. The generalist gets one head per task on a shared backbone (the
standard multi-task "one System 1" design) and may use a bigger backbone. We report per-task test objective,
parameters, and wall-clock latency of answering every task for one input.
"""
from __future__ import annotations

import copy
import json
import math
import random
import time
from typing import Any, Callable

import numpy as np
import torch

from . import jepa
from .config import Project, device as pick_device, find_workspace
from .data.store import Dataset
from .modalities import get_modality
from .model import SpecialistModel, count_params, task_loss
from .runtime import Pool, Specialist
from .spec import BACKBONES, TaskSpec
from .train import batches, evaluate_rows, targets_of, to_device


def train_generalist(specs: list[TaskSpec], backbone: str, log: Callable[[str], None] = print, seed: int = 0) -> tuple[SpecialistModel, dict[str, Any], TaskSpec]:
    torch.manual_seed(seed)
    rng = random.Random(seed)
    dev = pick_device()
    gspec = specs[0].model_copy(deep=True)
    gspec.name, gspec.backbone = "generalist", backbone
    # Use the most generous training budget among the specialists.
    gspec.train.epochs = max(s.train.epochs for s in specs)
    gspec.train.pretrain_steps = max(s.train.pretrain_steps for s in specs)
    gmod = get_modality(gspec)
    model = SpecialistModel(gmod.build_backbone(), {s.name: (len(s.labels), s.task_type) for s in specs})

    tasks = {}
    for s in specs:
        dd = Project(s.name).data_dir_for(s)
        mod = get_modality(s, str(dd))
        ds = Dataset(dd)
        tr, dv = ds.load("train"), ds.load("dev")
        tasks[s.name] = {"spec": s, "mod": mod, "train": [mod.load(r) for r in tr], "t": targets_of(s, mod, tr), "dev_rows": dv, "dev": [mod.load(r) for r in dv]}
    log(f"  generalist: {backbone} with {len(specs)} heads, {sum(len(t['train']) for t in tasks.values())} train examples on {dev}")

    if gspec.train.pretrain_steps > 0:
        unl = [x for t in tasks.values() for x in t["train"]]
        jepa.pretrain(model, gmod, unl, gspec.train.pretrain_steps, gspec.train.batch_size, gspec.train.lr_backbone, gspec.train.pretrain_lambda, dev, log, seed)

    model.to(dev)
    tc = gspec.train
    opt = torch.optim.AdamW([{"params": [p for h in model.heads.values() for p in h.parameters()], "lr": tc.lr_head},
                             {"params": list(model.backbone.parameters()), "lr": tc.lr_backbone}], weight_decay=0.01)
    plan = [(n, b) for n, t in tasks.items() for b in batches(len(t["train"]), tc.batch_size, True, rng)]
    steps = tc.epochs * len(plan)
    warm = max(1, steps // 15)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(s, steps) / steps)))
    best, best_score, thresholds = None, -1.0, {}
    for ep in range(tc.epochs):
        model.train()
        plan = [(n, b) for n, t in tasks.items() for b in batches(len(t["train"]), tc.batch_size, True, rng)]
        rng.shuffle(plan)  # interleave tasks
        for name, b in plan:
            t = tasks[name]
            s = t["spec"]
            fl = [t["t"][i] for i in b] if s.task_type == "frame" else None
            inputs, ft = gmod.collate([t["train"][i] for i in b], fl)
            logits, mask = model(to_device(inputs, dev), name)
            y = ft.to(dev) if ft is not None else torch.tensor([t["t"][i] for i in b], device=dev)
            loss = task_loss(logits, y, mask)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
        objs, thr = {}, {}
        for name, t in tasks.items():
            ev = evaluate_rows(model, gmod, t["spec"], t["dev_rows"], t["dev"], dev, None, calibrate=True, head=name)
            objs[name], thr[name] = ev["metrics"]["objective"], ev["threshold"]
        score = float(np.mean(list(objs.values())))
        log(f"  generalist epoch {ep + 1}/{tc.epochs} mean dev objective={score:.4f} " + " ".join(f"{k}={v:.3f}" for k, v in objs.items()))
        if score > best_score:
            best_score, thresholds = score, thr
            best = copy.deepcopy({k: v.cpu() for k, v in model.state_dict().items()})
    model.load_state_dict(best)
    return model, thresholds, gspec


def run(names: list[str], generalist_backbone: str | None = None, log: Callable[[str], None] = print) -> dict[str, Any]:
    specs = [TaskSpec.load(Project(n).best_dir / "spec.yaml") for n in names]
    mods = {s.modality for s in specs}
    if len(mods) != 1:
        raise ValueError(f"bench compares specialists of one modality; got {mods}")
    size = {c["id"]: c["params_m"] for cs in BACKBONES.values() for c in cs}
    bb = generalist_backbone or max((s.backbone for s in specs), key=lambda b: size.get(b, 0))
    model, thresholds, gspec = train_generalist(specs, bb, log)
    gmod = get_modality(gspec)
    dev = pick_device()
    rows = []
    for s in specs:
        p = Project(s.name)
        dd = p.data_dir_for(s)
        mod = get_modality(s, str(dd))
        test_rows = Dataset(dd).load("test")
        raws = [mod.load(r) for r in test_rows]
        g = evaluate_rows(model, gmod, s, test_rows, raws, dev, thresholds[s.name], head=s.name)["metrics"]
        # Same scorer for the specialist (builtin suite = its own held-out test, dev-calibrated threshold).
        sm = evaluate_rows(Specialist(p.best_dir).model, mod, s, test_rows, raws, dev, json.loads((p.best_dir / "meta.json").read_text())["threshold"])["metrics"]
        rows.append({"task": s.name, "metric": s.objective.metric, "specialist": sm["objective"], "generalist": g["objective"],
                     "specialist_macro_f1": sm["macro_f1"], "generalist_macro_f1": g["macro_f1"], "specialist_params": json.loads((p.best_dir / "meta.json").read_text())["params"]})

    # Latency to answer *every* task for one input: N specialists in parallel vs one shared forward pass + N heads.
    pool = Pool.load(names, device="cpu")
    x = next(iter(pool.specialists.values())).mod.example_input()
    for _ in range(5):
        pool.broadcast(x)
    ts = []
    for _ in range(30):
        t = time.perf_counter()
        pool.broadcast(x)
        ts.append((time.perf_counter() - t) * 1000)
    model.to("cpu").eval()
    inputs, _ = gmod.collate([x])
    gts = []
    with torch.no_grad():
        for _ in range(35):
            t = time.perf_counter()
            feats, mask = model.backbone(**inputs)
            for h in model.heads.values():
                h(feats, mask)
            gts.append((time.perf_counter() - t) * 1000)
    out = {
        "tasks": rows,
        "generalist_backbone": bb,
        "generalist_params": count_params(model.backbone) + sum(count_params(h) for h in model.heads.values()),
        "specialists_total_params": sum(r["specialist_params"] for r in rows),
        "latency_all_tasks_ms": {"specialists_parallel_p50": float(np.median(ts)), "generalist_p50": float(np.median(gts[5:]))},
        "mean_objective": {"specialists": float(np.mean([r["specialist"] for r in rows])), "generalist": float(np.mean([r["generalist"] for r in rows]))},
    }
    (find_workspace() / "bench").mkdir(exist_ok=True)
    (find_workspace() / "bench" / f"{'+'.join(names)}.json").write_text(json.dumps(out, indent=2))
    return out
