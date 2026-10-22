import json

from byosm.config import Project
from byosm.next_steps import suggestions
from byosm.spec import Label, TaskSpec


def _specialist(ws, n_test, n_calib, rivals=("jev",)):
    p = Project("qa").ensure()
    spec = TaskSpec(
        name="qa",
        description="is it a question",
        modality="text",
        labels=[Label(name="statement", definition="no"), Label(name="question", definition="yes")],
        baselines=list(rivals),
    )
    spec.save(p.spec_path)
    p.exp_dir("v1").mkdir(parents=True)
    (p.exp_dir("v1") / "meta.json").write_text("{}")
    p.promote("v1")
    p.board_path.write_text(json.dumps({"v1": {"builtin": {"objective": 0.9, "n_test": n_test, "n_calib": n_calib}}}))
    return p


def test_a_tiny_held_out_set_comes_first(ws):
    _specialist(ws, n_test=5, n_calib=18)
    first, *rest = suggestions("qa")
    assert first["args"] == ["build", "qa", "--force-data"] and "5 rows" in first["why"] and rest


def test_every_untried_proposal_is_its_own_step(ws):
    p = _specialist(ws, n_test=500, n_calib=200, rivals=())
    proposals = [{"id": "more-data", "why": "a"}, {"id": "regularize", "why": "b"}]
    (p.root / "proposals.json").write_text(json.dumps({"from": "v1", "proposals": proposals}))
    (p.exp_dir("v1") / "proposal.json").write_text(json.dumps({"applied": [{"id": "regularize"}]}))
    assert [s["id"] for s in suggestions("qa") if s["id"].startswith("try-")] == ["try-more-data"]


def test_unscored_rivals_a_missing_overfit_check_and_a_stale_export_are_suggested_in_order(ws):
    _specialist(ws, n_test=500, n_calib=200)
    ids = [s["id"] for s in suggestions("qa")]
    assert ids == ["score-rivals", "overfit", "propose", "export"]


def test_a_specialist_with_no_model_is_built_first(ws):
    TaskSpec(name="qa", description="q", modality="text", labels=[Label(name="a", definition="a")]).save(
        Project("qa").ensure().spec_path
    )
    assert [s["id"] for s in suggestions("qa")] == ["build"]
