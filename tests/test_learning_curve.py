import math

import pytest

from byosm.curve import CURVE, learning_curve, slope
from byosm.data.store import Dataset, record
from byosm.project import Project
from byosm.rungs import auxiliary
from byosm.spec import Label, TaskSpec, TrainCfg


def _project():
    p = Project("tc").ensure()
    labels = [Label(name=n, definition="") for n in "ab"]
    p.save_spec(TaskSpec(name="tc", description="", modality="text", labels=labels))
    return p


def test_the_gain_at_twice_the_data_follows_the_score_per_doubling():
    gain = slope([(100, 0.70), (200, 0.75), (400, 0.80)]) * math.log(2)
    assert gain == pytest.approx(0.05)


def test_a_fraction_of_train_is_a_nested_subset(tmp_path):
    from byosm.train.rows import prepare

    ds = Dataset(tmp_path)
    ds.save("train", [record(f"row {i}", "a", "t") for i in range(40)])
    ds.save("dev", [record("dev", "a", "t")])
    spec = TaskSpec(name="t", description="", modality="text", labels=[Label(name="a", definition="")])
    ids = {}
    for f in (0.25, 0.5):
        spec.train = TrainCfg(data_fraction=f)
        ids[f] = {r["id"] for r in prepare(spec, tmp_path, lambda _: None, 0, ["train"]).train}
    assert len(ids[0.25]) == 10 and len(ids[0.5]) == 20 and ids[0.25] < ids[0.5]


@pytest.mark.parametrize(("scores", "limited"), [((0.70, 0.75), True), ((0.799, 0.80), False)])
def test_more_data_is_proposed_only_while_the_curve_still_climbs(ws, monkeypatch, scores, limited):
    from conftest import save_board

    from byosm import pipeline
    from byosm.improve.rules_data import more_data

    p = _project()
    p.save_experiment("v1", meta={"n_train": 400, "calib_held": True})
    p.promote("v1", "test")
    save_board(p, {"v1": {"builtin": {"kind": "experiment", "objective": 0.8, "calib_objective": 0.8}}})
    fake = dict(zip(("v1-n25", "v1-n50"), zip((100, 200), scores, strict=True), strict=True))

    def build(name, exp, log, promote=True):
        assert not promote and auxiliary(exp)
        n, score = fake[exp]
        p.save_experiment(exp, meta={"n_train": n})
        save_board(p, {exp: {"builtin": {"kind": "experiment", "objective": score, "calib_objective": score}}})

    monkeypatch.setattr(pipeline, "build", build)
    report = learning_curve("tc", lambda _: None)
    assert report["data_limited"] is limited and p.report(CURVE)["experiment"] == "v1"
    spec = p.spec()
    spec.data.synthetic = True
    dg = {"spec": spec, "primary": "builtin", "suites": {"builtin": {"objective": 0.8}}, "curve": [0.7, 0.8]}
    dg |= {"late_best": True, "n_train": 400, "learning_curve": report}
    assert bool(more_data(dg)) is limited
