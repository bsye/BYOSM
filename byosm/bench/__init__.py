from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import numpy as np

from ..config import Project, find_workspace
from ..config import device as pick_device
from ..data.store import Dataset
from ..modalities import get_modality
from ..model import SpecialistModel, count_params
from ..runtime import Specialist
from ..spec import BACKBONES, TaskSpec
from ..train import evaluate_rows
from .generalist import train_generalist
from .latency import latencies


def _task_row(s: TaskSpec, model: SpecialistModel, gmod: Any, threshold: Any, dev: str) -> dict[str, Any]:
    p = Project(s.name)
    dd = p.data_dir_for(s)
    mod = get_modality(s, str(dd))
    test_rows = Dataset(dd).load("test")
    raws = [mod.load(r) for r in test_rows]
    g = evaluate_rows(model, gmod, s, test_rows, raws, dev, threshold, head=s.name)["metrics"]
    meta = p.meta(p.best_dir.name)
    specialist = Specialist(p.best_dir).model
    sm = evaluate_rows(specialist, mod, s, test_rows, raws, dev, meta["threshold"])["metrics"]
    return {
        "task": s.name,
        "metric": s.objective.metric,
        "specialist": sm["objective"],
        "generalist": g["objective"],
        "specialist_macro_f1": sm["macro_f1"],
        "generalist_macro_f1": g["macro_f1"],
        "specialist_params": meta["params"],
    }


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
    rows = [_task_row(s, model, gmod, thresholds[s.name], dev) for s in specs]
    out = {
        "tasks": rows,
        "generalist_backbone": bb,
        "generalist_params": count_params(model.backbone) + sum(count_params(h) for h in model.heads.values()),
        "specialists_total_params": sum(r["specialist_params"] for r in rows),
        "latency_all_tasks_ms": latencies(names, model, gmod),
        "mean_objective": {
            "specialists": float(np.mean([r["specialist"] for r in rows])),
            "generalist": float(np.mean([r["generalist"] for r in rows])),
        },
    }
    (find_workspace() / "bench").mkdir(exist_ok=True)
    (find_workspace() / "bench" / f"{'+'.join(names)}.json").write_text(json.dumps(out, indent=2))
    return out
