from conftest import promoted_specialist, save_board, with_train_rows

from byosm.next_steps import suggestions
from byosm.project import Project
from byosm.spec import Label, TaskSpec


def test_a_tiny_held_out_set_comes_first(ws):
    promoted_specialist(n_test=5, n_calib=18)
    first, *rest = suggestions("qa")
    assert first["args"] == ["build", "qa", "--force-data"] and "5 rows" in first["why"] and rest


def test_the_untried_proposals_form_one_experiments_step(ws):
    p = promoted_specialist(n_test=500, n_calib=200, rivals=())
    proposals = [
        {"id": "more-data", "why": "a", "ops": [{"op": "set", "path": "data.train_size", "value": 4000}]},
        {"id": "regularize", "why": "b", "ops": [{"op": "set", "path": "train.ema", "value": True}]},
    ]
    p.save_report("proposals", {"from": "v1", "proposals": proposals})
    p.save_experiment("v1", proposal={"applied": [proposals[1]]})
    step = next(s for s in suggestions("qa") if s["id"] == "experiments")
    assert step["experiments"] == ["more-data"] and "--each" in step["args"]


def test_unscored_rivals_a_missing_overfit_check_one_suite_and_a_stale_export_are_suggested_in_order(ws):
    promoted_specialist(n_test=500, n_calib=200)
    ids = [s["id"] for s in suggestions("qa")]
    assert ids == ["score-rivals", "propose", "overfit", "robust", "seeds", "export"]


def test_a_specialist_with_no_model_is_built_first(ws):
    Project("qa").save_spec(
        TaskSpec(name="qa", description="q", modality="text", labels=[Label(name="a", definition="a")])
    )
    assert [s["id"] for s in suggestions("qa")] == ["build"]


def test_too_few_rows_per_label_is_too_small_and_asks_for_more_data(ws):
    from byosm.data.sufficiency import TOO_SMALL, data_check

    p = promoted_specialist(n_test=500, n_calib=200)
    with_train_rows(p, 10)
    check = data_check(p, p.spec())
    assert check["verdict"] == TOO_SMALL and check["labels"]["question"]["rows"] == 10
    assert next(s for s in suggestions("qa") if s["id"] == "find-data")["args"] == ["datasets", "qa"]
    found = [{"dataset": "org/more", "role": "labeled", "why": "fits", "args": ["exp", "create", "qa", "try-more"]}]
    p.save_report("data-suggestions", found)
    step = next(s for s in suggestions("qa") if s["id"] == "more-data")
    assert step["title"] == "Train with org/more" and step["args"][:2] == ["exp", "create"]


def test_a_bigger_held_out_set_from_an_experiment_is_offered_as_the_benchmark(ws):
    from byosm.data.store import Dataset, record
    from byosm.spec import DataPlan

    p = promoted_specialist(n_test=5, n_calib=18)
    big = p.spec().model_copy(update={"data": DataPlan(train_size=999)})
    p.save_experiment("v2", spec=big, meta={})
    rows = [record(f"x{i}", "question", "t") for i in range(120)]
    for split in ("test", "calib"):
        Dataset(p.data_dir_for(big)).save(split, rows)
    first = suggestions("qa")[0]
    assert first["args"] == ["adopt-data", "qa", "v2"] and "120 test and 120 calib" in first["why"]


def test_no_experiment_data_is_offered_when_the_own_held_out_set_is_big_enough(ws):
    from byosm.data.store import Dataset, record
    from byosm.held_out import bigger_held_out

    p = promoted_specialist(n_test=500, n_calib=200)
    rows = [record(f"x{i}", "question", "t") for i in range(120)]
    for split in ("test", "calib"):
        Dataset(p.data_dir_for(p.spec())).save(split, rows)
    assert bigger_held_out(p) is None


def test_a_step_gains_the_health_its_checks_are_missing():
    from byosm.next_steps import _with_gain

    cs = [
        {"id": "data", "title": "Enough data", "weight": 1, "status": "fail"},
        {"id": "weakest", "title": "No weak label", "weight": 1, "status": "warn"},
        {"id": "rival", "title": "Beats the best rival", "weight": 2, "status": "ok"},
    ]
    step = _with_gain({"id": "find-data", "title": "t", "why": "w", "args": []}, cs)
    assert step["gain"] == 38 and step["unblocks"] == ["Enough data"]


def test_a_tool_choice_specialist_is_offered_the_hard_public_benchmarks_it_does_not_have(ws):
    from byosm.spec import EvalSuite

    p = promoted_specialist(n_test=500, n_calib=200)
    spec = p.spec()
    spec.labels = [Label(name="no_fit", definition="no"), Label(name="fits", definition="yes")]
    spec.positive_label = "fits"
    p.save_spec(spec)
    step = next(s for s in suggestions("qa") if s["id"] == "public-suites")
    assert step["args"] == ["suite", "public", "qa"] and "bfcl" in step["why"] and "when2call" in step["why"]
    spec.evals += [EvalSuite(name=n, kind="local") for n in ("bfcl", "bfcl-live", "when2call")]
    p.save_spec(spec)
    step = next(s for s in suggestions("qa") if s["id"] == "public-suites")
    assert step["args"] == ["eval", "qa", "--suite", "bfcl", "--suite", "bfcl-live", "--suite", "when2call"]
    save_board(p, {"v1": {n: {"kind": "experiment", "objective": 0.5} for n in ("bfcl", "bfcl-live", "when2call")}})
    assert all(s["id"] != "public-suites" for s in suggestions("qa"))
