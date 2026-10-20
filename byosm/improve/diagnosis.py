from __future__ import annotations

import json
from collections import Counter
from typing import Any

from ..config import Project
from ..spec import TaskSpec


def _confusions(p: Project, exp: str, suite: str, spec: TaskSpec, k: int = 6) -> list[tuple[str, str, int]]:
    f = p.exp_dir(exp) / "results" / f"{suite}.json"
    cm = json.loads(f.read_text())["metrics"].get("confusion") if f.exists() else None
    names = spec.label_names
    if cm:
        pairs = [(names[i], names[j], c) for i, row in enumerate(cm) for j, c in enumerate(row) if i != j and c]
    else:
        ef = p.exp_dir(exp) / "dev_errors.json"
        errs = json.loads(ef.read_text()) if ef.exists() else []
        counts = Counter((e["gold"], e["pred"]) for e in errs if e.get("gold") != "frames")
        pairs = [(g, pr, c) for (g, pr), c in counts.items()]
    return sorted(pairs, key=lambda t: -t[2])[:k]


def _scored_experiment(p: Project, board: dict) -> str:
    exp = p.promoted or next((k for k, v in board.items() if next(iter(v.values()))["kind"] == "experiment"), None)
    if not exp or exp not in board:
        raise RuntimeError("no scored experiment yet - run `byosm build` / `byosm eval` first")
    return exp


def _suite_summary(p: Project, exp: str, spec: TaskSpec, board: dict, s: str, r: dict, binary: bool) -> dict:
    bl = {k: v[s]["objective"] for k, v in board.items() if s in v and v[s]["kind"] == "baseline"}
    m = r["metrics"]
    weakest = sorted(((n, v["f1"]) for n, v in m.get("per_label", {}).items()), key=lambda t: t[1])[:3]
    return {
        "objective": r["objective"],
        "accuracy": m.get("accuracy"),
        "macro_f1": m.get("macro_f1"),
        "roc_auc": m.get("roc_auc"),
        "precision": m.get("precision") if binary else None,
        "recall": m.get("recall") if binary else None,
        "threshold": m.get("threshold"),
        "best_baseline": max(bl.items(), key=lambda kv: kv[1]) if bl else None,
        "weakest": weakest,
        "confusions": _confusions(p, exp, s, spec),
    }


def _converging_late(curve: list[float], best_ep: int, epochs: int) -> bool:
    return len(curve) >= 4 and len(curve) >= epochs and best_ep >= int(0.75 * len(curve))


def _training_summary(meta: dict, spec: TaskSpec) -> dict[str, Any]:
    hist = meta.get("history", [])
    curve = [h["dev_objective"] for h in hist]
    best_ep = meta.get("best_epoch", len(curve) - 1)
    return {
        "curve": curve,
        "losses": [h["loss"] for h in hist],
        "best_epoch": best_ep,
        "train_seconds": meta.get("train_seconds"),
        "epochs_run": len(curve),
        "epochs_cfg": spec.train.epochs,
        "n_train": meta.get("n_train"),
        "late_best": _converging_late(curve, best_ep, spec.train.epochs),
    }


def diagnose(name: str) -> dict[str, Any]:
    p = Project(name)
    base = TaskSpec.load(p.spec_path)
    board = p.board()
    exp = _scored_experiment(p, board)
    spec = TaskSpec.load(p.exp_dir(exp) / "spec.yaml")
    meta = json.loads((p.exp_dir(exp) / "meta.json").read_text())
    binary = spec.positive_label is not None or len(spec.labels) == 2
    suites = {s: _suite_summary(p, exp, spec, board, s, r, binary) for s, r in board[exp].items()}
    head = {
        "experiment": exp,
        "spec": spec,
        "base": base,
        "suites": suites,
        "dev": meta["dev"].get("objective"),
        "binary": binary,
    }
    return {**head, **_training_summary(meta, spec), "board": board, "primary": base.primary_suite()}
