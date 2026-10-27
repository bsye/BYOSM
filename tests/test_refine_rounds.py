from byosm import pipeline
from byosm.data.store import Dataset, refine_split, rounds_split
from byosm.project import WEIGHTS, Project
from byosm.spec import compile_spec


def _build(ws, monkeypatch, scores, rounds=2, seen=None):
    spec = compile_spec("isq", "detect questions")
    spec.train.refine_rounds = rounds
    p = Project("isq").ensure()
    p.save_spec(spec)
    scores, refined = iter(scores), []

    def fit(spec, data_dir, run_dir, log, splits):
        if seen is not None:
            seen.append({s: len(Dataset(data_dir).load(s)) for s in splits})
        score = next(scores)
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / WEIGHTS).write_text(str(score))
        return {"dev": {"objective": score}, "n_train": 1}, [score]

    def refine(spec, cfg, errors, data_dir, exp, log, split=None):
        refined.append(split)
        Dataset(data_dir).add(split, [{"text": "q?", "label": "yes"}])
        return {"added": 1}

    monkeypatch.setattr(pipeline, "ensure_data", lambda *a: ws)
    monkeypatch.setattr(pipeline, "fitted_max_len", lambda spec, dd, log: spec)
    monkeypatch.setattr(pipeline, "fit", fit)
    monkeypatch.setattr(pipeline, "refine", refine)
    monkeypatch.setattr(pipeline.evals, "run", lambda *a, **k: None)
    monkeypatch.setattr(pipeline, "bigger_held_out", lambda p: None)
    monkeypatch.setattr(pipeline, "maybe_promote", lambda p, exp, log: {})
    pipeline.build("isq", exp="v1", log=lambda _: None)
    return p, refined


def test_refining_stops_after_a_round_that_scores_worse_and_keeps_the_better_model(ws, monkeypatch):
    p, refined = _build(ws, monkeypatch, [0.87, 0.83, 0.85])
    run, meta = p.exp_dir("v1"), p.experiment("v1")["meta"]
    assert (run / WEIGHTS).read_text() == "0.87" and not (run / pipeline.KEPT_WEIGHTS).exists()
    assert meta["kept_round"] == 0 and len(meta["rounds"]) == 2 and refined == [rounds_split("v1")]


def test_a_build_refines_only_when_its_spec_asks_for_rounds(ws, monkeypatch):
    _, refined = _build(ws, monkeypatch, [0.5], rounds=0)
    assert refined == []


def test_a_retried_build_drops_the_rows_its_earlier_rounds_wrote_but_keeps_the_planned_ones(ws, monkeypatch):
    Dataset(ws).save(refine_split("v1"), [{"text": "planned?", "label": "yes"}])
    Dataset(ws).save(rounds_split("v1"), [{"text": f"old {i}?", "label": "yes"} for i in range(5)])
    seen = []
    _build(ws, monkeypatch, [0.5, 0.6], rounds=1, seen=seen)
    assert seen[0] == {"train": 0, refine_split("v1"): 1, rounds_split("v1"): 0}
    assert seen[1][rounds_split("v1")] == 1


def test_refine_writes_nothing_when_most_dev_errors_look_mislabelled(ws, monkeypatch):
    spec = compile_spec("isq", "detect questions")
    errors = [{"id": i, "gold": "yes", "pred": "no", "confidence": 0.9, "input": "q"} for i in range(4)]
    analysis = {"patterns": [], "suspect_gold_ids": ["0", "1", "3"]}
    monkeypatch.setattr(pipeline.synth_text, "refine_jobs", lambda *a: ([], analysis))
    monkeypatch.setattr(pipeline.synth_text, "generate", lambda *a, **k: [{"text": "x"}])
    out = pipeline.refine(spec, {}, errors, ws, "v1", lambda _: None)
    assert out["added"] == 0 and out["suspects"] == 3 and not Dataset(ws).load(rounds_split("v1"))
