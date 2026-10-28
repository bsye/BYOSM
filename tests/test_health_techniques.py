from byosm.improve.analyst import LLM_SYSTEM, _prompt
from byosm.improve.bandit import BLOCKER_BOOST, _boost
from byosm.improve.techniques import aimed_at, techniques, with_techniques
from byosm.spec import Label, TaskSpec


def _spec():
    return TaskSpec(name="t", description="", modality="text", labels=[Label(name=n, definition="") for n in "ab"])


def _family(fid):
    return next(t for t in techniques() if t.id == fid)


def test_a_family_aimed_at_a_failing_check_is_drawn_more_often():
    assert _boost(_family("label-noise"), set(), {"overfit"}) == BLOCKER_BOOST
    assert _boost(_family("label-noise"), set(), {"speed"}) == 1


def test_each_check_names_the_families_aimed_at_it_and_the_proposer_sees_health_and_targets():
    robust = aimed_at(_spec(), "robust")
    assert robust and all("robust" in t.targets for t in robust)
    [c] = with_techniques(_spec(), [{"id": "overfit", "status": "fail"}])
    assert c["techniques"] and len(c["techniques"]) <= 2
    health = [{"id": "robust", "title": "Holds on new inputs", "status": "fail", "value": "-0.19", "why": "w"}]
    dg = {"spec": _spec(), "board": {}, "curve": [], "suites": {}, "history": [], "health": health, "next_steps": []}
    prompt = _prompt(dg)
    assert "Holds on new inputs" in prompt and '"targets": [' in prompt
    assert "clear every health check that is not ok" in LLM_SYSTEM


def test_find_more_experiments_says_which_failing_checks_it_aims_at(ws):
    from conftest import promoted_specialist

    from byosm.next_steps import suggestions

    promoted_specialist(n_test=500, n_calib=200)
    step = next(s for s in suggestions("qa") if s["id"] in ("propose", "experiments"))
    assert "Aims first at" in step["why"]


def test_proposals_record_the_failing_checks_their_family_targets(ws, monkeypatch):
    from byosm.improve import proposals as props
    from byosm.project import Project

    p = Project("a").ensure()
    p.save_spec(_spec())
    dg = {"spec": p.spec(), "experiment": None, "train_seconds": 60}
    ops = [{"op": "set", "path": "data.clean_labels", "value": True}]
    monkeypatch.setattr(props, "diagnose", lambda name: dg)
    monkeypatch.setattr(props, "rule_proposals", lambda dg: [])
    monkeypatch.setattr(
        props, "llm_proposals", lambda dg, log: [{"id": "llm-x", "why": "w", "ops": ops, "source": "llm"}]
    )
    monkeypatch.setattr(props, "failing_checks", lambda p: {"overfit", "speed"})
    _, kept = props.propose("a", log=lambda _: None)
    assert kept[0]["family"] == "label-noise" and kept[0]["aims"] == ["overfit"]
