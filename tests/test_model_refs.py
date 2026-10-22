import json

from byosm.model_refs import model_ref
from byosm.project import Project
from byosm.spec import Label, TaskSpec


def test_models_resolve_to_a_page_and_what_they_are(ws):
    p = Project("qa", ws)
    p.save_spec(TaskSpec(name="qa", description="q", modality="text", labels=[Label(name="a", definition="a")]))
    p.save_experiment("v1", meta={})
    p.promote("v1")
    llm = {
        "identity": json.dumps({"provider": "openai", "model": "qwen3.8:27b", "base_url": "http://localhost:11434/v1"})
    }
    assert model_ref(p, "llm", llm)["url"] == "https://ollama.com/library/qwen3.8"
    assert model_ref(p, "hf:org/model#A=b")["url"] == "https://huggingface.co/org/model"
    assert model_ref(p, "livekit-turn")["url"] == "https://huggingface.co/livekit/turn-detector"
    assert model_ref(p, "v1") == {"id": "v1", "name": "v1", "detail": "this model, not published", "url": None}


def test_an_added_llm_rival_is_a_named_role_a_baseline_and_a_test_model(ws):
    from byosm.config import load_config
    from byosm.evals.workbench import chosen_models
    from byosm.llm_rivals import add_llm_rival

    p = Project("qa", ws)
    p.save_spec(TaskSpec(name="qa", description="q", modality="text", labels=[Label(name="a", definition="a")]))
    pid = add_llm_rival(p, {"provider": "claude-code", "model": "claude-sonnet-5-5"})
    assert pid == "llm:rival-claude-sonnet-5-5" and pid in p.spec().baselines
    assert load_config(ws)["llm"]["rival-claude-sonnet-5-5"]["model"] == "claude-sonnet-5-5"
    assert pid in chosen_models(p, p.spec())
    assert model_ref(p, pid)["name"] == "claude-sonnet-5-5"
