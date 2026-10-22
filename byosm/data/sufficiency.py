from __future__ import annotations

from typing import Any

from ..project import Project
from ..spec import TaskSpec
from .request import counts
from .store import Dataset

MIN_TEST_ITEMS = 100
MIN_CALIB_ITEMS = 50
TOO_SMALL, THIN, ENOUGH = "too small", "thin", "enough"
SHORT_SHARE = 0.25


def _trained_on(p: Project, spec: TaskSpec) -> TaskSpec:
    return p.exp_spec(p.promoted) if p.promoted else spec


def data_check(p: Project, spec: TaskSpec) -> dict[str, Any] | None:
    if spec.task_type != "classify":
        return None
    dd = p.data_dir_for(_trained_on(p, spec))
    have = Dataset(dd).summary()["train"]["labels"] if Dataset(dd).built() else None
    if have is None and not p.promoted:
        return None
    need = counts(spec, []) if have is not None else {}
    labels = {
        n: {"rows": (have or {}).get(n, 0), "min": c["train_min"], "recommended": c["train_recommended"]}
        for n, c in need.items()
    }
    held = p.board().get(p.promoted or "", {}).get(spec.primary_suite(), {})
    n_test, n_calib = held.get("n_test") or 0, held.get("n_calib") or 0
    short = [n for n, x in labels.items() if x["rows"] < x["min"]]
    thin = [n for n, x in labels.items() if x["rows"] < x["recommended"]]
    small_held = bool(held) and (n_test < MIN_TEST_ITEMS or n_calib < MIN_CALIB_ITEMS)
    mostly_short = len(short) >= SHORT_SHARE * max(1, len(labels)) and bool(short)
    verdict = TOO_SMALL if mostly_short or small_held else THIN if thin else ENOUGH
    return {
        "verdict": verdict,
        "train": sum(x["rows"] for x in labels.values()),
        "labels": labels,
        "short": short,
        "thin": thin,
        "test": n_test,
        "calib": n_calib,
        "min_test": MIN_TEST_ITEMS,
        "min_calib": MIN_CALIB_ITEMS,
    }
