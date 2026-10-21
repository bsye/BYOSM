import json

import pytest

from byosm import llm
from byosm.auto import run
from byosm.config import Project
from byosm.discover import discover
from byosm.spec import TaskSpec, compile_spec


@pytest.mark.slow
def test_autopilot_end_to_end(ws):
    llm._cache.clear()
    p = Project("auto")
    spec = compile_spec("auto", "detect questions")
    spec.data.train_size, spec.data.dev_size, spec.data.test_size, spec.data.per_call = 120, 40, 60, 20
    spec.train.epochs, spec.train.pretrain_steps, spec.train.refine_rounds = 2, 0, 0
    spec.objective.target = 0.9
    spec, _ = discover(spec, {}, None, lambda m: None)
    p.ensure()
    spec.save(p.spec_path)
    final = run("auto", budget_h=1, max_experiments=4, max_params_m=1.0, echo=lambda m: None)
    report = (p.root / "auto.md").read_text()
    steps = [d["step"] for d in json.loads((p.root / "auto.json").read_text())["decisions"]]
    assert "llm" in TaskSpec.load(p.spec_path).baselines
    assert final["won"] and final["ours"] > final["best_baseline"][1]
    assert "distill" in steps and "overfit" in steps and "export" in steps and "WON" in report
    assert json.loads((p.root / "overfit.json").read_text())["experiment"] == final["exp"]
    assert (p.root / "export" / "manifest.json").exists()
