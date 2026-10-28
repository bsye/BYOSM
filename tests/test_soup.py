import pytest
import torch

from byosm.project import WEIGHTS, Project
from byosm.replicas import recipe_of, recipe_runs
from byosm.spec import Label, TaskSpec

HELD = {"calib_held": True, "params": 1}


def _result(calib):
    return {"kind": "experiment", "objective": calib, "calib_objective": calib, "fingerprint": "f"}


def _seeded(ws):
    from conftest import save_board

    p = Project("tc").ensure()
    p.save_spec(
        TaskSpec(name="tc", description="", modality="text", labels=[Label(name=n, definition="") for n in "ab"])
    )
    for i, e in enumerate(("v2", "v2-s1", "v2-s2")):
        p.save_experiment(e, meta=HELD)
        p.exp_dir(e).mkdir(parents=True, exist_ok=True)
        torch.save({"w": torch.tensor([1.0, 2.0]) * (i + 1), "ids": torch.tensor([7, 8])}, p.exp_dir(e) / WEIGHTS)
        save_board(p, {e: {"builtin": _result(0.80 + i / 100)}})
    return p


def _scored(p, calib):
    from conftest import save_board

    def run(name, experiments, log):
        save_board(p, {e: {"builtin": _result(calib)} for e in experiments})

    return run


def test_a_soup_averages_the_float_weights_of_every_clean_seed_and_keeps_integer_buffers(ws, monkeypatch):
    from byosm import evals
    from byosm.soup import make_soup

    p = _seeded(ws)
    monkeypatch.setattr(evals, "run", _scored(p, 0.9))
    soup = make_soup("tc", "v2", lambda _: None)
    state = torch.load(p.exp_dir(soup) / WEIGHTS)
    assert soup == "v2-soup" and torch.allclose(state["w"], torch.tensor([2.0, 4.0]))
    assert state["ids"].tolist() == [7, 8]
    assert recipe_of(soup) == "v2" and soup not in recipe_runs(p, soup)
    assert p.meta(soup)["soup_of"] == ["v2", "v2-s1", "v2-s2"]


@pytest.mark.parametrize(("calib", "ships"), [(0.9, "v2-soup"), (0.7, "v2")])
def test_the_soup_ships_only_when_it_scores_at_least_the_seed_mean_on_calib(ws, monkeypatch, calib, ships):
    from byosm import evals
    from byosm.replicas import runs
    from byosm.soup import shipped

    p = _seeded(ws)
    monkeypatch.setattr(evals, "run", _scored(p, calib))
    assert shipped(p, "v2", runs(p, "v2", "builtin"), lambda _: None) == ships
