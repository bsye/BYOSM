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
