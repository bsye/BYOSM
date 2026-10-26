from byosm import pipeline
from byosm.project import WEIGHTS, Project
from byosm.spec import compile_spec


def test_a_refine_round_that_scores_worse_on_dev_does_not_replace_the_better_model(ws, monkeypatch):
    spec = compile_spec("isq", "detect questions")
    spec.train.refine_rounds = 2
    p = Project("isq").ensure()
    p.save_spec(spec)
    scores = iter([0.87, 0.83, 0.85])

    def fit(spec, data_dir, run_dir, log, splits):
        score = next(scores)
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / WEIGHTS).write_text(str(score))
        return {"dev": {"objective": score}, "n_train": 1}, [score]

    monkeypatch.setattr(pipeline, "ensure_data", lambda *a: ws)
    monkeypatch.setattr(pipeline, "fitted_max_len", lambda spec, dd, log: spec)
    monkeypatch.setattr(pipeline, "fit", fit)
    monkeypatch.setattr(pipeline, "refine", lambda *a: {"added": 1})
    monkeypatch.setattr(pipeline.evals, "run", lambda *a, **k: None)
    monkeypatch.setattr(pipeline, "bigger_held_out", lambda p: None)
    monkeypatch.setattr(pipeline, "maybe_promote", lambda p, exp, log: {})
    pipeline.build("isq", log=lambda _: None)
    run = p.exp_dir("v1")
    meta = p.experiment("v1")["meta"]
    assert (run / WEIGHTS).read_text() == "0.87" and not (run / pipeline.KEPT_WEIGHTS).exists()
    assert meta["kept_round"] == 0 and meta["dev"]["objective"] == 0.87 and len(meta["rounds"]) == 3
