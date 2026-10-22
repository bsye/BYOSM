import pytest

from byosm.data.store import ShingleIndex, write
from byosm.evals import overfit, overfit_checks
from byosm.project import Project
from byosm.spec import TaskSpec
from byosm.spec.sections import EvalSuite, Label, Objective

TRAIN_TEXTS = [f"the {w} order number {i} arrived late today" for i, w in enumerate(["red", "blue", "green", "pink"])]
FRESH_TEXTS = [f"a brand new sentence about topic {i} nobody wrote before" for i in range(8)]
HISTORY = [(1.0, 0.70), (0.6, 0.80), (0.4, 0.85)]
TRAIN_SCORE = 0.99


def _units(correct):
    return {"y": [1] * len(correct), "pred": [1 if c else 0 for c in correct]}


def _result(test_correct, calib_correct, ci):
    return {
        "kind": "experiment",
        "objective": sum(test_correct) / len(test_correct),
        "calib_objective": sum(calib_correct) / len(calib_correct),
        "metrics": {"ci95": ci},
        "units": {"test": _units(test_correct), "calib": _units(calib_correct)},
    }


class Scorer:
    def __init__(self):
        self.calls = 0

    def __call__(self, pred, spec, calib, test, root, space=None):
        self.calls += 1
        return {"objective": TRAIN_SCORE, "metrics": {"ci95": {"lo": 0.98, "hi": 1.0}}}


@pytest.fixture
def toy(ws, monkeypatch):
    p = Project("toy", ws).ensure()
    spec = TaskSpec(
        name="toy",
        description="toy",
        modality="text",
        labels=[Label(name="no", definition="no"), Label(name="yes", definition="yes")],
        objective=Objective(metric="accuracy"),
        evals=[EvalSuite(name="bench", kind="local", primary=True)],
    )
    spec.train.epochs = len(HISTORY)
    p.save_spec(spec)
    dd = p.data_dir_for(spec)
    write(dd / "train.jsonl", [{"id": str(i), "input": t, "label": "yes"} for i, t in enumerate(TRAIN_TEXTS)])
    write(dd / "dev.jsonl", [{"id": "d", "input": "dev row", "label": "yes"}])
    test = [{"id": f"t{i}", "input": t, "label": "yes"} for i, t in enumerate(TRAIN_TEXTS[:2] + FRESH_TEXTS)]
    write(p.evals_dir / "bench" / "test.jsonl", test)
    write(p.evals_dir / "bench" / "calib.jsonl", test[:4])
    for exp in ("v1", "v2"):
        p.exp_dir(exp).mkdir(parents=True)
        p.save_experiment(exp, spec=spec)
    history = [{"epoch": i, "loss": loss, "dev_objective": d} for i, (loss, d) in enumerate(HISTORY)]
    meta = {"dev": {"objective": 0.85}, "best_epoch": 2, "history": history}
    p.save_experiment("v1", meta=meta)
    p.save_experiment("v2", meta=meta)
    board = {
        "v1": {"bench": _result([True, True] + [True] * 4 + [False] * 4, [True] * 4, {"lo": 0.55, "hi": 0.65})},
        "v2": {"bench": _result([False] * 10, [False] * 4, {"lo": 0.0, "hi": 0.0})},
    }
    p.save_board(board)
    p.promote("v1")
    scorer = Scorer()
    monkeypatch.setattr(overfit, "score", scorer)
    monkeypatch.setattr(overfit, "experiment_predictor", lambda p, base, exp: exp)
    return p, scorer, p.spec()


def _by_check(report):
    return {f["check"]: f for f in report["findings"]}


def test_the_check_finds_memorisation_leakage_and_calib_overuse(toy):
    p, scorer, _ = toy
    report = overfit.check("toy", ws=p.ws, log=lambda m: None)
    found = _by_check(report)
    assert report["experiment"] == "v1" and report["verdict"] == "fail"
    assert report["scores"] == {"train": TRAIN_SCORE, "dev": 0.85, "calib": 1.0, "test": 0.6}
    assert found["gap"]["severity"] == "fail" and found["gap"]["numbers"]["gap"] == 0.39
    assert found["drift"]["severity"] == "fail"
    leak = found["overlap"]["numbers"]
    assert found["overlap"]["severity"] == "fail" and (leak["exact"], leak["near"]) == (2, 0)
    assert leak["test_without_overlap"] == 0.5 and leak["same_label"] == 2
    assert found["curve"]["severity"] == "warn" and "still rising" in found["curve"]["explanation"]
    assert p.report("overfit")["verdict"] == "fail"


def test_the_check_is_cached_and_reuses_the_train_score_when_only_the_board_changes(toy):
    p, scorer, _ = toy
    overfit.check("toy", ws=p.ws, log=lambda m: None)
    overfit.check("toy", ws=p.ws, log=lambda m: None)
    assert scorer.calls == 1
    board = p.board()
    board["v2"]["bench"]["calib_objective"] = 0.99
    p.save_board(board)
    again = overfit.check("toy", ws=p.ws, log=lambda m: None)
    assert scorer.calls == 1 and _by_check(again)["selection"]["numbers"]["median_others"] == 0.99
    p.save_experiment("v1", meta={**p.meta("v1"), "train_seconds": 1})
    overfit.check("toy", ws=p.ws, log=lambda m: None)
    assert scorer.calls == 2


def test_the_command_prints_every_finding_and_the_verdict(toy):
    from typer.testing import CliRunner

    from byosm.cli import app

    out = CliRunner().invoke(app, ["overfit", "toy"], env={"COLUMNS": "200"})
    assert out.exit_code == 0, out.output
    assert "Generalisation gap" in out.output and "Selection pressure" in out.output
    assert "fail: Overfitting or leakage found." in out.output


def test_a_check_needs_a_scored_experiment(toy):
    p, _, _ = toy
    with pytest.raises(RuntimeError, match="no bench result"):
        overfit.check("toy", exp="v3", ws=p.ws, log=lambda m: None)


def test_a_late_dev_drop_while_loss_falls_is_overfitting():
    f = overfit_checks.curve([0.80, 0.90, 0.85, 0.84], [1.0, 0.5, 0.3, 0.2], 1, 8)
    assert f["severity"] == "warn" and "overfit" in f["explanation"]
    assert overfit_checks.curve([0.80, 0.90, 0.90], [1.0, 0.5, 0.4], 1, 8)["severity"] == "ok"


def test_an_overlap_that_does_not_move_the_score_is_only_a_warning(toy):
    _, _, spec = toy
    rows = [{"input": t, "label": "yes"} for t in TRAIN_TEXTS[:2] + FRESH_TEXTS]
    units = _units([False, False] + [True] * 4 + [False] * 4)
    f = overfit_checks.overlap("bench", ShingleIndex(rows[:2]), rows, spec, units)
    assert f["severity"] == "warn" and f["numbers"]["test_without_overlap"] == 0.5


def test_many_equal_experiments_make_the_calib_winner_look_better_by_chance():
    ci = {"lo": 0.90, "hi": 0.96}
    lucky = overfit_checks.selection(0.94, [0.935] * 29, ci)
    assert lucky["severity"] == "warn" and lucky["numbers"]["experiments"] == 30
    assert overfit_checks.selection(0.94, [0.80, 0.82], ci)["severity"] == "ok"
    assert overfit_checks.expected_lift(1, 0.01) == 0.0
