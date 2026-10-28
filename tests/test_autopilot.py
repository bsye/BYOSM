import pytest

from byosm import llm
from byosm.auto import run
from byosm.discover import discover
from byosm.project import Project
from byosm.spec import compile_spec


@pytest.mark.slow
def test_autopilot_end_to_end(ws):
    llm.clear_cache()
    p = Project("auto")
    spec = compile_spec("auto", "detect questions")
    spec.data.train_size, spec.data.dev_size, spec.data.test_size, spec.data.per_call = 120, 40, 60, 20
    spec.train.epochs, spec.train.pretrain_steps, spec.train.refine_rounds = 2, 0, 0
    spec.objective.target, spec.finalist_seeds = 0.9, 1
    spec, _ = discover(spec, {}, None, lambda m: None)
    p.ensure()
    p.save_spec(spec)
    final = run("auto", budget_h=1, max_experiments=4, max_params_m=1.0, echo=lambda m: None)
    report = p.report("autopilot-summary")
    steps = [d["step"] for d in p.report("autopilot")["decisions"]]
    assert "llm" in p.spec().baselines
    assert final["won"] and final["ours"] > final["best_baseline"][1]
    assert "distill" in steps and "overfit" in steps and "export" in steps and "WON" in report
    assert p.report("overfit")["experiment"] == final["exp"]
    assert (p.root / "export" / "manifest.json").exists()
