from byosm.improve.bandit import VERDICT, family_order, rank, search_summary
from byosm.improve.portfolio import recipes
from byosm.project import Project
from byosm.spec import Label, TaskSpec

EPOCHS = {"op": "set", "path": "train.epochs", "value": 12}


def _specialist(name, modality="text"):
    p = Project(name).ensure()
    labels = [Label(name=n, definition="") for n in "ab"]
    p.save_spec(TaskSpec(name=name, description="", modality=modality, labels=labels))
    return p


def _tried(p, exp, ops, won, hours=0.5):
    p.save_experiment(exp, meta={}, proposal={"parent": "v1", "applied": [{"id": exp, "why": "w", "ops": ops}]})
    p.save_report(VERDICT + exp, {"won": won, "gpu_hours": hours})


def test_untried_families_come_first_and_one_that_failed_three_times_drops_out(ws):
    p = _specialist("a")
    for i in range(3):
        _tried(p, f"v{i + 2}", [{"op": "set", "path": "train.epochs", "value": 10 + i * 30}], False)
    order = family_order(p, p.spec())
    assert "schedule" not in order
    assert order.index("real-data") < len(order) - 1


def test_ranking_takes_one_proposal_per_family_before_a_second_from_any(ws):
    p = _specialist("a")
    props = [
        {"id": "s1", "family": "schedule", "probability": 0.9},
        {"id": "s2", "family": "schedule", "probability": 0.8},
        {"id": "h1", "family": "head", "probability": 0.1},
    ]
    ids = [pr["id"] for pr in rank(p, p.spec(), props)]
    assert set(ids[:2]) == {"s1", "h1"} and ids[2] == "s2"


def test_a_recipe_that_won_on_another_specialist_of_the_same_kind_is_offered_first(ws):
    mine, other, audio = _specialist("a"), _specialist("b"), _specialist("c", "audio")
    data = {"op": "set", "path": "data.train_size", "value": 9000}
    _tried(other, "v2", [EPOCHS, data], True)
    _tried(audio, "v2", [{"op": "set", "path": "train.lr_head", "value": 0.01}], True)
    found = recipes(mine, mine.spec())
    assert [r["ops"] for r in found] == [[EPOCHS]]
    assert "b" in found[0]["why"] and found[0]["source"] == "portfolio"


def test_the_search_summary_counts_families_repeats_and_gpu_hours_per_sure_lead(ws):
    p = _specialist("a")
    _tried(p, "v2", [EPOCHS], True, 1.0)
    _tried(p, "v3", [EPOCHS], False, 1.0)
    _tried(p, "v4", [{"op": "set", "path": "train.ema", "value": True}], False, 2.0)
    s = search_summary(p)
    assert s["experiments"] == 3 and s["sure_leads"] == 1 and s["repeats"] == 1
    assert s["gpu_hours_per_sure_lead"] == 4.0
    assert {f["family"]: f["tries"] for f in s["families"]} == {"schedule": 2, "regularisation": 1}


def test_every_promotion_vote_leaves_a_verdict_for_the_bandit(ws):
    from conftest import paired, save_board

    from byosm.promotion import maybe_promote

    p = _specialist("a")
    p.save_spec(p.spec().model_copy(update={"finalist_seeds": 1}))
    p.save_experiment("v1", meta={"calib_held": True})
    p.promote("v1", "test")
    p.save_experiment(
        "v2", meta={"calib_held": True, "gpu_hours": 0.25}, proposal={"applied": [{"id": "x", "ops": [EPOCHS]}]}
    )
    result = {"kind": "experiment", "n_test": 500, "n_calib": 200}
    save_board(
        p,
        {
            "v1": {"builtin": result | {"objective": 0.8, "calib_objective": 0.8} | paired(0.8)},
            "v2": {"builtin": result | {"objective": 0.9, "calib_objective": 0.9} | paired(0.9)},
        },
    )
    maybe_promote(p, "v2", lambda _: None)
    v = p.report(VERDICT + "v2")
    assert v["won"] and v["family"] == "schedule" and v["gpu_hours"] == 0.25


def test_proposals_that_change_nothing_or_restate_an_earlier_idea_are_dropped(ws, monkeypatch):
    from byosm.improve import proposals as props

    p = _specialist("a")
    why = "dev peaked early and the loss kept falling so the encoder overfits the small training set badly"
    p.save_experiment("v2", meta={}, proposal={"applied": [{"id": "x", "why": why, "ops": [EPOCHS]}]})
    dg = {"spec": p.spec(), "experiment": None, "train_seconds": 60}
    same = {"op": "set", "path": "train.epochs", "value": p.spec().train.epochs}
    llm = [
        {"id": "llm-a", "why": why, "ops": [{"op": "set", "path": "train.ema", "value": True}], "source": "llm"},
        {
            "id": "llm-b",
            "why": "pool tokens by attention",
            "ops": [{"op": "set", "path": "train.pool", "value": "attention"}],
            "source": "llm",
        },
        {"id": "llm-c", "why": "no change", "ops": [same], "source": "llm"},
    ]
    monkeypatch.setattr(props, "diagnose", lambda name: dg)
    monkeypatch.setattr(props, "rule_proposals", lambda dg: [])
    monkeypatch.setattr(props, "llm_proposals", lambda dg, log: llm)
    _, kept = props.propose("a", log=lambda _: None)
    assert [pr["id"] for pr in kept] == ["llm-b"] and kept[0]["family"] == "head"


def test_the_techniques_page_lists_every_family_with_its_tries_and_sure_leads_across_specialists(ws):
    from fastapi.testclient import TestClient

    from byosm.app import create_app

    a, b = _specialist("a"), _specialist("b")
    _tried(a, "v2", [EPOCHS], True, 1.0)
    _tried(b, "v2", [EPOCHS], False, 0.5)
    with TestClient(create_app(ws, "t"), base_url="http://127.0.0.1") as c:
        rows = c.get("/api/techniques", headers={"Authorization": "Bearer t"}).json()
    by_id = {r["id"]: r for r in rows}
    assert by_id["schedule"]["tries"] == 2 and by_id["schedule"]["wins"] == 1 and by_id["schedule"]["specialists"] == 2
    assert by_id["schedule"]["gpu_hours"] == 1.5 and by_id["setfit"]["tries"] == 0
    assert {r["status"] for r in rows} == {"available", "built-in", "manual"}
