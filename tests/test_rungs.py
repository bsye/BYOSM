from byosm.project import Project
from byosm.rungs import auxiliary, screen, short
from byosm.spec import Label, TaskSpec

SCORES = {"v2-r1": 0.7, "v3-r1": 0.9, "v4-r1": 0.8, "v5-r1": 0.6}


def _specialist(eta=3):
    p = Project("tc").ensure()
    labels = [Label(name=n, definition="") for n in "ab"]
    p.save_spec(TaskSpec(name="tc", description="", modality="text", labels=labels, screen_eta=eta))
    return p


def _fake_build(p, built):
    from conftest import save_board

    def build(name, exp, log, promote=True):
        assert not promote
        built.append((exp, p.exp_spec(exp).train.epochs))
        if exp == "v5-r1":
            raise RuntimeError("out of memory")
        p.save_experiment(exp, meta={"calib_held": True})
        save_board(p, {exp: {"builtin": {"kind": "experiment", "objective": 0.5, "calib_objective": SCORES[exp]}}})

    return build


def test_a_short_run_trains_a_third_of_the_epochs_without_refine_rounds():
    spec = TaskSpec(name="t", description="", modality="text", labels=[Label(name="a", definition="")])
    spec.train.epochs, spec.train.refine_rounds = 8, 2
    s = short(spec)
    assert s.train.epochs == 3 and s.train.refine_rounds == 0 and spec.train.epochs == 8


def test_screening_keeps_the_best_third_on_calib_and_skips_a_run_that_failed(ws, monkeypatch):
    from byosm import pipeline

    p = _specialist()
    for e in ("v2", "v3", "v4", "v5"):
        p.save_experiment(e, proposal={"applied": []})
    built = []
    monkeypatch.setattr(pipeline, "build", _fake_build(p, built))
    assert screen("tc", ["v2", "v3", "v4", "v5"], lambda _: None) == ["v3", "v4"]
    assert [e for e, _ in built] == ["v2-r1", "v3-r1", "v4-r1", "v5-r1"]
    assert auxiliary("v3-r1") and not auxiliary("v3")


def test_screening_is_off_for_one_candidate_or_an_eta_of_one(ws):
    _specialist(eta=1)
    assert screen("tc", ["v2", "v3"], lambda _: None) == ["v2", "v3"]
    _specialist()
    assert screen("tc", ["v2"], lambda _: None) == ["v2"]
