import pytest

from byosm import improve
from byosm.events import EVENTS_ENV, parse
from byosm.export import export
from byosm.pipeline import build
from byosm.project import Project
from byosm.spec import compile_spec


def quiet(_):
    return None


@pytest.mark.slow
def test_text_pipeline_end_to_end(ws, load_runtime, monkeypatch, capfd):
    monkeypatch.setenv(EVENTS_ENV, "1")
    spec = compile_spec("isq", "detect questions")
    spec.data.dev_size, spec.data.per_call = 40, 20
    spec.train.epochs, spec.train.pretrain_steps, spec.train.refine_rounds = 3, 10, 1
    spec.baselines, spec.finalist_seeds = ["llm"], 1
    p = Project("isq").ensure()
    p.save_spec(spec)
    build("isq", log=quiet)
    assert p.promoted == "v1"
    board = p.board()
    assert board["v1"]["builtin"]["objective"] > 0.9 and any(k.startswith("llm") for k in board)
    events = [e for e in map(parse, capfd.readouterr().out.splitlines()) if e]
    epochs = [e for e in events if e["kind"] == "epoch"]
    assert epochs[0]["epoch"] == 1 and epochs[0]["epochs"] == 3 and epochs[0]["experiment"] == "v1"
    scored = [(e["predictor"], e["suite"]) for e in events if e["kind"] == "scored"]
    assert ("v1", "builtin") in scored

    _, _ = improve.propose("isq", use_llm=False, log=quiet)
    proposal = {"id": "x", "why": "", "ops": [{"op": "set", "path": "train.epochs", "value": 2}]}
    p.save_report("proposals", {"from": "v1", "proposals": [proposal]})
    exp = improve.apply("isq", ["x"])
    assert exp == "v2"
    build("isq", exp, log=quiet)
    assert set(p.experiments()) == {"v1", "v2"} and len(list((p.root / "data").iterdir())) == 1

    m = load_runtime(export("isq", log=quiet))
    assert m.predict("what about the coffee?")["label"] == "question"
    assert m.encode("what about the coffee?").shape == (256,)
