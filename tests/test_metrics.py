import numpy as np

from byosm.metrics import _cms, bootstrap, calibrate, compute, objective_from_cm, units
from byosm.spec import Label, Objective, TaskSpec


def labels(*names):
    return [Label(name=n, definition="") for n in names]


def test_threshold_calibration_meets_precision_constraint():
    spec = TaskSpec(
        name="t",
        description="",
        modality="text",
        labels=labels("neg", "pos"),
        positive_label="pos",
        objective=Objective(metric="recall_at_precision", constraint=0.95),
    )
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 2000)
    score = np.clip(y * 0.35 + rng.normal(0.35, 0.15, 2000), 0, 1)
    probs = np.stack([1 - score, score], 1)
    thr = calibrate(spec, y, probs)
    m = compute(spec, y, probs, thr)
    assert m["precision"] >= 0.95 and m["recall"] > 0.3


def test_paired_bootstrap():
    spec = TaskSpec(
        name="t", description="", modality="text", labels=labels("a", "b", "c"), objective=Objective(metric="accuracy")
    )
    rng = np.random.default_rng(0)
    y = rng.integers(0, 3, 600)
    good = np.where(rng.random(600) < 0.9, y, (y + 1) % 3)
    bad = np.where(rng.random(600) < 0.8, y, (y + 1) % 3)
    a, b = units(3, y, good), units(3, y, bad)
    assert bootstrap(spec, a, b)["p_better"] > 0.99 and bootstrap(spec, b, a)["p_better"] < 0.01
    assert abs(bootstrap(spec, a, a)["p_better"] - 0.5) < 1e-9
    ci = bootstrap(spec, a)
    assert ci["lo"] < (good == y).mean() < ci["hi"]


def test_frame_units_are_files_with_the_flat_objective():
    spec = TaskSpec(
        name="t", description="", modality="text", labels=labels("a", "b", "c"), objective=Objective(metric="accuracy")
    )
    rng = np.random.default_rng(0)
    y = rng.integers(0, 3, 600)
    good = np.where(rng.random(600) < 0.9, y, (y + 1) % 3)
    fu = units(3, y, good, groups=[100, 200, 300])
    assert len(fu["cm"]) == 3 and abs(objective_from_cm(spec, _cms(fu, 3).sum(0)) - (good == y).mean()) < 1e-9
