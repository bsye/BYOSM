import json

import pytest

from byosm import improve
from byosm.config import Project
from byosm.export import export
from byosm.pipeline import build
from byosm.spec import compile_spec


def quiet(_):
    return None


@pytest.mark.slow
def test_text_pipeline_end_to_end(ws, load_runtime):
    spec = compile_spec("isq", "detect questions")
    spec.data.dev_size, spec.data.per_call = 40, 20
    spec.train.epochs, spec.train.pretrain_steps, spec.train.refine_rounds = 3, 10, 1
    spec.baselines = ["llm"]
    p = Project("isq").ensure()
    spec.save(p.spec_path)
    build("isq", log=quiet)
    assert p.promoted == "v1"
    board = p.board()
    assert board["v1"]["builtin"]["objective"] > 0.9 and any(k.startswith("llm") for k in board)

    _, _ = improve.propose("isq", use_llm=False, log=quiet)
    proposal = {"id": "x", "why": "", "ops": [{"op": "set", "path": "train.epochs", "value": 2}]}
    (p.root / "proposals.json").write_text(json.dumps({"from": "v1", "proposals": [proposal]}))
    exp = improve.apply("isq", ["x"])
    assert exp == "v2"
    build("isq", exp, log=quiet)
    assert set(p.experiments()) == {"v1", "v2"} and len(list((p.root / "data").iterdir())) == 1

    m = load_runtime(export("isq", log=quiet))
    assert m.predict("what about the coffee?")["label"] == "question"
    assert m.encode("what about the coffee?").shape == (256,)
