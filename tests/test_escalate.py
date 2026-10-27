from test_proposals import LOST, make_project, session

from byosm.auto import escalate
from byosm.health import _target
from byosm.spec.backbones import MINILM

MBERT = "hf:google-bert/bert-base-multilingual-cased"


def test_the_next_rung_is_the_smallest_larger_backbone_not_tried_yet(ws):
    p, spec = make_project(ws)
    s = session(p, spec)
    assert p.exp_spec("v1").backbone == MINILM
    assert escalate.bigger_backbone(s) == "hf:distilbert/distilbert-base-cased"
    p.save_experiment("v2", spec=spec.model_copy(update={"backbone": "hf:distilbert/distilbert-base-cased"}))
    assert escalate.bigger_backbone(s) == "hf:microsoft/deberta-v3-xsmall"


def test_a_non_english_specialist_only_climbs_to_backbones_that_speak_its_language(ws):
    p, spec = make_project(ws)
    p.save_experiment("v1", spec=spec.model_copy(update={"languages": ["es"]}))
    assert (
        escalate.bigger_backbone(session(p, spec)) == "hf:sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )


def test_with_no_larger_backbone_left_the_target_is_marked_unreachable_with_what_was_tried(ws, monkeypatch):
    p, spec = make_project(ws)
    p.save_experiment("v1", spec=spec.model_copy(update={"backbone": MBERT}))
    monkeypatch.setattr(escalate, "standing", lambda name, margin: {**LOST, "target": 0.9})
    reason = escalate.escalate(session(p, spec), "no untried proposals left")
    assert reason.startswith("unreachable at this budget")
    target = _target(p, spec, {"objective": 0.5})
    assert "unreachable at this budget" in target["value"] and MBERT in target["why"]
