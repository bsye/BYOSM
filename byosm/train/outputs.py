from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from ..data.store import CALIB, Dataset
from ..db.rows import exists
from ..project import WEIGHTS
from ..spec import TaskSpec
from .common import cpu_state


def snapshot(model: torch.nn.Module, ep: int, ev: dict[str, Any]) -> dict[str, Any]:
    m = ev["metrics"]
    return {
        "objective": m["objective"],
        "macro_f1": m["macro_f1"],
        "epoch": ep,
        "threshold": ev["threshold"],
        "state": cpu_state(model),
        "dev": ev,
    }


def build_meta(
    spec: TaskSpec,
    best: dict[str, Any],
    history: list[dict],
    jepa_hist: list[dict],
    n_params: int,
    dev: str,
    seconds: float,
    counts: dict[str, int],
) -> dict[str, Any]:
    return {
        "name": spec.name,
        "labels": spec.label_names,
        "threshold": best["threshold"],
        "best_epoch": best["epoch"],
        "dev": best["dev"]["metrics"],
        "params": n_params,
        "device": dev,
        "train_seconds": seconds,
        "n_train": counts["n_train"],
        "n_dev": counts["n_dev"],
        "history": history,
        "jepa": jepa_hist[:: max(1, len(jepa_hist) // 50)],
        "distilled_from": spec.train.distill_from,
        "n_teacher_only": counts["n_teacher_only"],
        "n_test_folded": counts["n_test_folded"],
    }


def save_run(run_dir: Path, best: dict[str, Any]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    torch.save(best["state"], run_dir / WEIGHTS)


def finish(
    run_dir: Path, ds: Dataset, splits: tuple[str, ...] | list[str], meta: dict[str, Any], best: dict[str, Any]
) -> tuple[dict[str, Any], list[Any]]:
    meta["train_splits"] = list(splits)
    meta["calib_held"] = exists(ds.path(CALIB))
    save_run(run_dir, best)
    return meta, json.loads(json.dumps(best["dev"]["errors"], default=str))
