from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from ..spec import TaskSpec


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


def save_run(spec: TaskSpec, run_dir: Path, best: dict[str, Any], meta: dict[str, Any]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    torch.save(best["state"], run_dir / "model.pt")
    spec.save(run_dir / "spec.yaml")
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    errors = json.dumps(best["dev"]["errors"], indent=2, ensure_ascii=False, default=str)
    (run_dir / "dev_errors.json").write_text(errors)
