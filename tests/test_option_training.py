from option_pool import pool

from byosm.data import build
from byosm.data.store import CALIB, Dataset
from byosm.data.table import read_table
from byosm.options.create import pair_plan, spec_for
from byosm.options.pairs import write

SOURCE = "lockon/xlam-function-calling-60k"
QUIET = {"log": lambda m: None}


def _built(ws, spec, folder):
    build(spec, ws / folder, {"concurrency": 1}, **QUIET)
    return {s: Dataset(ws / folder).load(s) for s in ("train", "dev", CALIB, "test")}


def _table(ws, plan):
    table = ws / "pairs.csv"
    write(table, pool(300), plan)
    return table, read_table(table)


def test_the_base_plan_trains_on_the_table_and_keeps_its_marked_calib_and_test(ws):
    plan = pair_plan(held_out=True)
    table, rows = _table(ws, plan)
    got = _built(ws, spec_for("tc", table, SOURCE, plan), "base")
    texts = {s: {r["text"] for r in rows if r["split"] == s} for s in got}
    assert sorted(r["input"] for r in got["train"]) == sorted(r["text"] for r in rows if r["split"] == "train")
    for s in ("dev", CALIB, "test"):
        assert got[s] and {r["input"] for r in got[s]} <= texts[s]
    assert {r["meta"]["group"] for r in got["train"]}.isdisjoint({r["meta"]["group"] for r in got[CALIB]})


def test_an_experiment_changes_how_train_is_written_but_not_calib_or_test(ws):
    plan = pair_plan(held_out=True)
    table, _ = _table(ws, plan)
    spec = spec_for("tc", table, SOURCE, plan)
    exp = spec.model_copy(deep=True)
    exp.data.pairs.layouts, exp.data.pairs.mask_names = ["json", "openai"], 0.5
    base, varied = _built(ws, spec, "base"), _built(ws, exp, "varied")
    assert {r["meta"]["layout"] for r in varied["train"]} == {"json", "openai"}
    masked = [r for r in varied["train"] if "tool_1" in r["input"]]
    assert 0.3 < len(masked) / len(varied["train"]) < 0.7
    for s in (CALIB, "test"):
        assert sorted(r["input"] for r in varied[s]) == sorted(r["input"] for r in base[s])


def test_a_table_without_a_held_out_plan_has_calib_carved_from_dev_by_request(ws):
    plan = pair_plan(held_out=False)
    table, _ = _table(ws, plan)
    got = _built(ws, spec_for("tc", table, SOURCE, plan), "base")
    assert got["dev"] and got[CALIB]
    assert {r["meta"]["group"] for r in got["dev"]}.isdisjoint({r["meta"]["group"] for r in got[CALIB]})
