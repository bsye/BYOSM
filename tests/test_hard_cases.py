import numpy as np
import pytest
from conftest import save_board

import byosm.hard_cases as hard
from byosm import pools
from byosm.benchmarks.pairs import Case
from byosm.benchmarks.when2call import called
from byosm.data.store import Dataset, refine_split
from byosm.options.situations import FITS, NO_FIT, Situation
from byosm.project import Project
from byosm.spec import Label, TaskSpec

WEATHER = {"name": "weather", "description": "Weather for a city", "parameters": {"city": {"required": True}}}
NEWS = {"name": "news", "description": "Headlines on a topic", "parameters": {"topic": {"required": True}}}
HELD = "what is the weather in paris tomorrow morning please"


def _cases():
    yield Case("a", "tool_call", Situation("show me the weather in rome this weekend", [WEATHER, NEWS], {"weather"}))
    yield Case("b", "no_call", Situation("what is the weather like", [WEATHER], set()))
    yield Case("c", "no_call", Situation(HELD, [WEATHER], set()))


def _specialist(ws, monkeypatch):
    monkeypatch.setattr(pools, "POOLS", {"fake": (lambda spec: True, _cases, "org/fake")})
    labels = [Label(name=FITS, definition="call it now"), Label(name=NO_FIT, definition="do not call it now")]
    spec = TaskSpec(name="tc", description="", modality="text", labels=labels, positive_label=FITS)
    p = Project("tc").ensure()
    p.save_spec(spec)
    data = Dataset(p.data_dir_for(spec))
    data.save("calib", [{"input": f"{HELD}\n\nOption: weather", "label": NO_FIT}])
    data.save("test", [{"input": "find news on football", "label": FITS}])
    p.save_experiment("v1", meta={"calib_held": True})
    p.promote("v1", "test")
    return p


def test_a_reply_names_the_tools_it_calls_and_a_question_calls_none():
    assert called('<TOOLCALL>[{"name": "weather", "arguments": {"city": "Rome"}}]</TOOLCALL>') == {"weather"}
    assert called("To proceed, could you tell me which city?") == set()


def test_the_pool_leaves_out_requests_that_copy_a_calib_or_test_row(ws, monkeypatch):
    p = _specialist(ws, monkeypatch)
    rows = hard.pool_rows(p, p.spec(), "fake", lambda _: None)
    assert {r["meta"]["group"] for r in rows} == {"a", "b"}
    assert sorted((r["meta"]["group"], r["label"]) for r in rows) == [("a", FITS), ("a", NO_FIT), ("b", NO_FIT)]


def test_mining_keeps_the_rows_the_model_gets_most_wrong(ws, monkeypatch):
    p = _specialist(ws, monkeypatch)
    rows = hard.pool_rows(p, p.spec(), "fake", lambda _: None)
    monkeypatch.setattr(hard, "HARD_ROWS", 2)
    monkeypatch.setattr(hard, "_losses", lambda p, base, parent, sample, log: np.array([0.1, 2.0, 0.9]))
    kept = hard.mine(p, p.spec(), "v1", rows, lambda _: None)
    sample = hard._sample(rows)
    assert kept == [sample[1], sample[2]]


def test_a_round_trains_warm_on_the_earlier_rows_plus_the_new_ones(ws, monkeypatch):
    from byosm import pipeline

    p = _specialist(ws, monkeypatch)
    data = Dataset(p.data_dir_for(p.spec()))
    data.save(refine_split("v1"), [{"input": "an earlier hard case", "label": NO_FIT}])
    monkeypatch.setattr(hard, "mine", lambda p, base, parent, rows, log: rows[:1])
    built = []
    monkeypatch.setattr(pipeline, "build", lambda name, exp, log: built.append(exp) or {})
    hard.hard_round("tc", lambda _: None)
    [exp] = built
    added = [r["input"] for r in data.load(refine_split(exp))]
    assert added[0] == "an earlier hard case" and len(added) == 2
    assert p.exp_spec(exp).train.init_from == "v1"
    assert (p.proposal(exp) or {})["applied"][0]["id"] == hard.HARD_STEP


def test_a_round_that_lost_is_not_run_again_from_the_same_model(ws, monkeypatch):
    from byosm.improve.bandit import VERDICT

    p = _specialist(ws, monkeypatch)
    p.save_experiment("v2", proposal={"parent": "v1", "applied": [{"id": hard.HARD_STEP, "why": "", "ops": []}]})
    p.save_report(VERDICT + "v2", {"won": False})
    with pytest.raises(RuntimeError, match="did not win"):
        hard.hard_round("tc", lambda _: None)


def test_seed_replicas_train_on_the_rows_their_recipe_added(ws, monkeypatch):
    from byosm import pipeline
    from byosm.replicas import replicate

    p = _specialist(ws, monkeypatch)
    data = Dataset(p.data_dir_for(p.spec()))
    data.save(refine_split("v1"), [{"input": "a hard case", "label": NO_FIT}])
    save_board(p, {"v1": {"builtin": {"kind": "experiment", "objective": 0.9, "calib_objective": 0.9}}})
    monkeypatch.setattr(pipeline, "build", lambda name, exp, log, promote=True, suites=None: {})
    made = replicate("tc", "v1", 2, lambda _: None)
    assert made and all(data.load(refine_split(e)) == data.load(refine_split("v1")) for e in made)


def test_the_robustness_check_asks_for_a_hard_case_round_until_one_from_this_model_lost(ws, monkeypatch):
    from byosm.evals.robust import _fix
    from byosm.improve.bandit import VERDICT

    p = _specialist(ws, monkeypatch)
    assert _fix(p)[0] == [hard.HARD_STEP, "tc"]
    p.save_experiment("v2", proposal={"parent": "v1", "applied": [{"id": hard.HARD_STEP, "why": "", "ops": []}]})
    p.save_report(VERDICT + "v2", {"won": False})
    assert _fix(p)[0] == ["improve", "tc"]


def test_the_hard_case_step_is_a_command():
    from byosm.cli.base import app

    assert hard.HARD_STEP in {c.name for c in app.registered_commands}
