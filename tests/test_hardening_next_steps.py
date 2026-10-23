from conftest import save_board

from byosm.next_steps import suggestions
from byosm.project import Project
from byosm.spec import Label, TaskSpec


def _promoted(ws):
    p = Project("qa").ensure()
    labels = [Label(name="statement", definition="no"), Label(name="question", definition="yes")]
    p.save_spec(TaskSpec(name="qa", description="is it a question", modality="text", labels=labels, baselines=[]))
    for exp in ("v1", "v2"):
        p.exp_dir(exp).mkdir(parents=True)
    p.save_experiment("v1", meta={})
    old = {"id": "more-data", "why": "a", "ops": [{"path": "data.per_label", "value": 800}]}
    p.save_experiment("v2", meta={}, proposal={"parent": "v1", "applied": [old]})
    p.promote("v2")
    save_board(p, {"v2": {"builtin": {"kind": "experiment", "objective": 0.9, "n_test": 500, "n_calib": 200}}})
    return p, old


def test_a_used_proposal_id_with_new_ops_is_offered_again(ws):
    p, old = _promoted(ws)
    new = {"id": "more-data", "why": "a", "ops": [{"path": "data.per_label", "value": 3200}]}
    p.save_report("proposals", {"from": "v2", "proposals": [new]})
    step = next(s for s in suggestions("qa") if s["id"] in ("experiments", "propose"))
    assert step["id"] == "experiments" and step["experiments"] == ["more-data"]


def test_a_proposal_with_the_same_ops_as_one_already_applied_is_not_offered(ws):
    p, old = _promoted(ws)
    p.save_report("proposals", {"from": "v2", "proposals": [dict(old)]})
    assert any(s["id"] == "propose" for s in suggestions("qa"))
