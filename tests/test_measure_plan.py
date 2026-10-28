import pytest
from conftest import promoted_specialist, with_train_rows

from byosm.archive import set_archived
from byosm.measure.plan import CPU, GPU, PRETRAINING, plan

PRETRAIN_V2 = ["series", "pretrain", "--recipe", "v2"]


def _cells(found):
    return [(c["lane"], c["specialist"], c["measurement"], c["args"]) for c in found]


def _forecaster(ws, domain="energy", **fit):
    from byosm.series.board import save_report
    from byosm.series.prediction import FitCfg, Options

    version = {"name": "v1", "recipe": "boosting", "effort": 0, **Options(fit=FitCfg(**fit)).model_dump()}
    version |= {"calib": 0.9, "test": 0.9, "crps": 0.5, "ci": None, "configs": {}}
    save_report(ws, {"domain": domain, "at": 1.0, "configs": {}, "versions": [version], "promoted": "v1"})


def test_cpu_checks_come_first_then_the_gpu_trainings_one_specialist_after_the_other(ws):
    promoted_specialist("qa")
    promoted_specialist("vad")
    assert _cells(plan(ws)) == [
        (CPU, "qa", "overfit", ["overfit", "qa"]),
        (CPU, "vad", "overfit", ["overfit", "vad"]),
        (GPU, "qa", "seeds", ["seeds", "qa"]),
        (GPU, "vad", "seeds", ["seeds", "vad"]),
    ]
    assert {c["model"] for c in plan(ws)} == {"v1"}


def test_a_measured_model_drops_out_and_a_stale_check_comes_back(ws):
    p = promoted_specialist()
    measured = {"verdict": "ok", "summary": "train and test agree", "findings": [{"check": "fresh"}]}
    p.save_report("overfit", {"experiment": "v1", "verdict": "ok", "summary": "train and test agree"})
    assert [c["measurement"] for c in plan(ws)] == ["overfit", "seeds"]
    p.save_report("overfit", {"experiment": "v1", **measured})
    assert [c["measurement"] for c in plan(ws)] == ["seeds"]
    p.save_report("overfit", {"experiment": "v0", "verdict": "ok", "summary": "an older model"})
    assert [c["measurement"] for c in plan(ws)] == ["overfit", "seeds"]


def test_public_benchmarks_and_the_learning_curve_join_when_they_fit(ws):
    p = promoted_specialist(labels=("no_fit", "fits"), positive="fits")
    with_train_rows(p, 10)
    found = {c["measurement"]: c for c in plan(ws)}
    assert found["public-suites"]["args"] == ["suite", "public", "qa"] and found["public-suites"]["lane"] == CPU
    assert found["curve"]["args"] == ["curve", "qa"]
    assert [c["measurement"] for c in plan(ws)] == ["overfit", "public-suites", "seeds", "curve"]


def test_a_forecaster_that_never_tried_v2_pretrains_its_base_then_runs_until_it_beats_the_rival(ws, monkeypatch):
    from byosm.series.patch import weights_path
    from byosm.series.recipe import V2

    monkeypatch.setenv("BYOSM_CACHE", str(ws / "cache"))
    _forecaster(ws)
    assert _cells(plan(ws)) == [(GPU, "energy", PRETRAINING, PRETRAIN_V2)]
    weights = weights_path(V2.weights("base"))
    weights.parent.mkdir(parents=True)
    weights.write_bytes(b"")
    assert plan(ws)[0]["args"] == ["series", "run", "energy", "--until-rival", "--rounds", "2"]
    _forecaster(ws, pretraining=V2.name)
    assert plan(ws) == []


def test_archived_specialists_are_left_out_and_unknown_names_refused(ws):
    promoted_specialist("qa")
    promoted_specialist("vad")
    set_archived(ws, "vad", True)
    assert {c["specialist"] for c in plan(ws)} == {"qa"}
    with pytest.raises(KeyError, match="vad"):
        plan(ws, ["vad"])


def test_a_specialist_whose_next_steps_fail_is_left_out(ws, monkeypatch):
    from byosm import next_steps

    promoted_specialist("qa")
    promoted_specialist("vad")
    real = next_steps.suggestions

    def broken(name, ws=None):
        if name == "vad":
            raise ValueError("unreadable spec")
        return real(name, ws)

    monkeypatch.setattr(next_steps, "suggestions", broken)
    assert {c["specialist"] for c in plan(ws)} == {"qa"}


def test_the_command_prints_the_plan_on_a_dry_run_and_queues_it_otherwise(ws):
    from typer.testing import CliRunner

    from byosm.cli import app
    from byosm.measure.run import saved_run

    promoted_specialist()
    assert CliRunner().invoke(app, ["measure"]).exit_code != 0
    dry = CliRunner().invoke(app, ["measure", "--all", "--dry-run"])
    assert dry.exit_code == 0 and "1. cpu  qa, Overfitting" in dry.output and "(byosm seeds qa)" in dry.output
    assert saved_run(ws)["cells"] == []
    out = CliRunner().invoke(app, ["measure", "qa"])
    assert out.exit_code == 0 and "queued 2 measurements; `byosm app` starts them" in out.output
    assert [c["measurement"] for c in saved_run(ws)["cells"]] == ["overfit", "seeds"]
    assert CliRunner().invoke(app, ["measure", "nope"]).exit_code != 0
