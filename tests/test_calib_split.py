from byosm.data.splits import carve_calib
from byosm.data.store import CALIB, record
from byosm.spec import Label, TaskSpec


def _rows(label, texts, **meta):
    return [record(t, label, "local", **meta) for t in texts]


def test_calib_takes_half_of_dev_with_every_label_on_both_sides():
    dev = _rows("a", [f"alpha request number {i} about cards" for i in range(40)])
    dev += _rows("b", [f"beta question {i} on transfers abroad" for i in range(40)])
    splits = {"dev": dev}
    carve_calib(splits)
    assert len(splits["dev"]) == len(splits[CALIB]) == 40
    assert {r["label"] for r in splits[CALIB]} == {r["label"] for r in splits["dev"]} == {"a", "b"}
    assert not {r["id"] for r in splits["dev"]} & {r["id"] for r in splits[CALIB]}


def test_near_duplicates_and_clips_of_one_recording_stay_on_the_same_side():
    twin = "please block my card it was stolen yesterday at the station"
    dev = _rows("a", [twin, twin + " today"]) + _rows("b", [f"unrelated text {i}" for i in range(6)])
    dev += _rows([0, 1], ["clip-1.wav", "clip-2.wav"], file="call.wav")
    splits = {"dev": dev}
    carve_calib(splits)
    for side in ("dev", CALIB):
        texts = {r["input"] for r in splits[side]}
        assert (twin in texts) == (twin + " today" in texts)
        assert ("clip-1.wav" in texts) == ("clip-2.wav" in texts)


def _spec():
    labels = [Label(name="a", definition=""), Label(name="b", definition="")]
    return TaskSpec(name="tc", description="", modality="text", labels=labels, finalist_seeds=1)


def test_a_model_trained_before_calib_was_held_out_cannot_win_promotion_on_the_builtin_suite(ws):
    from conftest import save_board

    from byosm.project import Project
    from byosm.promotion import maybe_promote

    p = Project("tc").ensure()
    p.save_spec(_spec())
    p.save_experiment("v1", meta={"calib_held": True})
    p.save_experiment("v2", meta={})
    p.save_experiment("v3", meta={"calib_held": True})
    p.promote("v1", "test")
    result = {"kind": "experiment", "n_test": 500, "n_calib": 200}
    save_board(
        p,
        {
            "v1": {"builtin": result | {"objective": 0.8, "calib_objective": 0.8}},
            "v2": {"builtin": result | {"objective": 0.99, "calib_objective": 0.99}},
            "v3": {"builtin": result | {"objective": 0.9, "calib_objective": 0.9}},
        },
    )
    said = []
    maybe_promote(p, "v2", said.append)
    assert p.promoted == "v1"
    assert "before calib was held out" in said[0]
    maybe_promote(p, "v3", said.append)
    assert p.promoted == "v3"


def test_builtin_suites_score_selection_on_calib_and_never_on_dev(ws):
    from byosm.data.store import Dataset
    from byosm.evals.suites import stored
    from byosm.project import Project

    p = Project("tc").ensure()
    spec = _spec()
    p.save_spec(spec)
    ds = Dataset(p.data_dir_for(spec))
    for split in ("train", "dev", CALIB, "test"):
        ds.save(split, _rows("a", [f"{split} row"]))
    calib, test, _ = stored(p, spec, spec.evals[0])
    assert [r["input"] for r in calib] == [f"{CALIB} row"]
    assert [r["input"] for r in test] == ["test row"]


def test_a_calib_the_source_already_marked_is_kept_and_dev_is_left_whole():
    splits = {"dev": _rows("a", ["one", "two"]), CALIB: _rows("a", ["three"])}
    carve_calib(splits)
    assert [r["input"] for r in splits["dev"]] == ["one", "two"]
    assert [r["input"] for r in splits[CALIB]] == ["three"]
