from __future__ import annotations

import statistics
from typing import Any

from .. import metrics
from ..data.store import NEAR_DUPLICATE, ShingleIndex
from ..spec import TaskSpec
from ..text import normalize

SEVERITIES = ("ok", "warn", "fail")
GAP_WARN = 0.05
GAP_FAIL = 0.15
DRIFT_WARN = 0.02
OVERLAP_WARN_SHARE = 0.005
OVERLAP_FAIL_SHARE = 0.02
OVERLAP_EXAMPLES = 3
OVERLAP_HARMLESS = 0.005
SELECTION_OPTIMISM_WARN = 0.01
CURVE_RISE = 0.002
CURVE_WINDOW = 3
CURVE_DROP = 0.01
Z_95 = 1.96
ROUND = 4
TITLES = {
    "gap": "Generalisation gap",
    "drift": "Calib vs test",
    "overlap": "Train/test overlap",
    "selection": "Selection pressure",
    "curve": "Learning curve",
}


def finding(check: str, severity: str, explanation: str, numbers: dict[str, Any], **extra: Any) -> dict[str, Any]:
    nums = {k: round(float(v), ROUND) if isinstance(v, float) else v for k, v in numbers.items()}
    head = {"check": check, "title": TITLES[check], "severity": severity, "explanation": explanation}
    return {**head, "numbers": nums, **extra}


def worst(findings: list[dict[str, Any]]) -> str:
    return max((f["severity"] for f in findings), key=SEVERITIES.index, default="ok")


def gap(train: float, dev: float | None, calib: float | None, test: float) -> dict[str, Any]:
    g = train - test
    sev = "fail" if g > GAP_FAIL else "warn" if g > GAP_WARN else "ok"
    text = {
        "ok": "The model scores about the same on rows it trained on as on the test, so it learned more than it memorised.",
        "warn": "The model scores clearly higher on rows it trained on than on the test: some memorisation.",
        "fail": "The model scores far higher on rows it trained on than on the test: it memorised its training data.",
    }[sev]
    nums = {"train": train, "dev": dev, "calib": calib, "test": test, "gap": g, "warn_above": GAP_WARN}
    return finding("gap", sev, text, nums)


def drift(suite: str, calib: float, calib_ci: dict, test: float, test_ci: dict) -> dict[str, Any]:
    d = calib - test
    if calib_ci["lo"] > test_ci["hi"]:
        sev, text = "fail", f"On {suite}, calib is clearly higher than test: calib was over-used to pick the model."
    elif test_ci["lo"] > calib_ci["hi"]:
        sev, text = "warn", f"On {suite}, test is clearly higher than calib: the two splits are not alike."
    elif d > DRIFT_WARN:
        sev, text = "warn", f"On {suite}, calib is higher than test, though within their confidence intervals."
    else:
        sev, text = "ok", f"On {suite}, calib and test agree within their confidence intervals."
    nums = {"calib": calib, "calib_lo": calib_ci["lo"], "calib_hi": calib_ci["hi"], "test": test}
    nums |= {"test_lo": test_ci["lo"], "test_hi": test_ci["hi"], "difference": d}
    return finding("drift", sev, text, nums, suite=suite)


def _matches(index: ShingleIndex, rows: list[dict]) -> list[dict]:
    hits = []
    for i, r in enumerate(rows):
        if not isinstance(r["input"], str):
            continue
        same = index.exact.get(normalize(r["input"]))
        near = (same, 1.0) if same is not None else index.nearest(r["input"])
        if near is not None and near[1] > NEAR_DUPLICATE:
            twin = index.rows[near[0]]
            hit = {"row": i, "test": r["input"], "train": twin["input"], "similarity": round(near[1], ROUND)}
            hits.append({**hit, "exact": same is not None, "same_label": twin["label"] == r["label"]})
    return hits


def _without(spec: TaskSpec, units: dict[str, list], rows: set[int]) -> float | None:
    kept = {k: [x for i, x in enumerate(v) if i not in rows] for k, v in units.items()}
    return metrics.objective_of_units(spec, kept) if all(kept.values()) else None


def _overlap_numbers(test_rows: list[dict], hits: list[dict], spec: TaskSpec, units: dict) -> dict[str, Any]:
    exact = sum(h["exact"] for h in hits)
    nums = {"test_rows": len(test_rows), "exact": exact, "near": len(hits) - exact, "share": len(hits) / len(test_rows)}
    nums["same_label"] = sum(h["same_label"] for h in hits)
    if units and all(len(v) == len(test_rows) for v in units.values()):
        nums["test"] = metrics.objective_of_units(spec, units)
        nums["test_without_overlap"] = _without(spec, units, {h["row"] for h in hits})
    return nums


def _overlap_verdict(suite: str, nums: dict[str, Any]) -> tuple[str, str]:
    clean = nums.get("test_without_overlap")
    inflation = nums["test"] - clean if clean is not None else None
    if nums["exact"] / nums["test_rows"] > OVERLAP_FAIL_SHARE:
        if inflation is not None and inflation <= OVERLAP_HARMLESS:
            return "warn", f"Many {suite} test inputs are also in training, but the score barely moves without them."
        return "fail", f"Many {suite} test inputs appear word for word in the training rows: the benchmark leaks."
    if nums["share"] > OVERLAP_WARN_SHARE:
        return "warn", f"Some {suite} test inputs also appear, exactly or nearly, in the training rows."
    return "ok", f"Almost no {suite} test input also appears in the training rows."


def overlap(suite: str, index: ShingleIndex, test_rows: list[dict], spec: TaskSpec, units: dict) -> dict[str, Any]:
    hits = _matches(index, test_rows)
    nums = _overlap_numbers(test_rows, hits, spec, units)
    sev, text = _overlap_verdict(suite, nums)
    return finding("overlap", sev, text, nums, suite=suite, examples=hits[:OVERLAP_EXAMPLES])


def expected_lift(n: int, calib_se: float) -> float:
    if n < 2:
        return 0.0
    return calib_se * statistics.NormalDist().inv_cdf((n - 0.375) / (n + 0.25))


def selection(promoted: float, others: list[float], calib_ci: dict) -> dict[str, Any]:
    n = len(others) + 1
    lift = expected_lift(n, (calib_ci["hi"] - calib_ci["lo"]) / (2 * Z_95))
    median = statistics.median(others) if others else None
    lead = promoted - median if median is not None else None
    luck = lift > SELECTION_OPTIMISM_WARN and (lead is None or lead < lift)
    if n < 2:
        text = "Only one experiment was compared on calib, so picking it added no optimism."
    elif luck:
        text = f"{n} experiments were compared on calib; chance alone could explain the winner's lead."
    else:
        text = f"{n} experiments were compared on calib; the winner leads by more than chance would give."
    nums = {"experiments": n, "promoted_calib": promoted, "median_others": median, "lead": lead, "chance_lift": lift}
    return finding("selection", "warn" if luck else "ok", text, nums)


def curve(dev: list[float], losses: list[float], best: int, epochs_cfg: int) -> dict[str, Any]:
    nums: dict[str, Any] = {"epochs_run": len(dev), "epochs_cfg": epochs_cfg, "best_epoch": best + 1}
    if not dev:
        return finding("curve", "ok", "No training history was recorded.", nums)
    nums |= {"best_dev": dev[best], "last_dev": dev[-1], "best_loss": losses[best], "last_loss": losses[-1]}
    window = dev[-CURVE_WINDOW:]
    rising = len(dev) >= epochs_cfg and best == len(dev) - 1 and window[-1] > window[0] + CURVE_RISE
    falling = dev[-1] < dev[best] - CURVE_DROP and losses[-1] < losses[best]
    if rising:
        return finding("curve", "warn", "Dev was still rising at the last epoch: more epochs would likely help.", nums)
    if falling:
        text = "Train loss kept falling while dev fell after the best epoch: later epochs overfit (the best was kept)."
        return finding("curve", "warn", text, nums)
    return finding("curve", "ok", "Dev levelled off before training ended, with no late drop.", nums)
