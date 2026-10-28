import random

from byosm.options.layouts import marked
from byosm.options.pairs import table_rows, write
from byosm.options.select import score
from byosm.options.situations import FITS, NO_FIT, Situation, with_unrelated, xlam
from byosm.spec import PairPlan

ROW = {
    "query": "weather in Paris and Rome?",
    "tools": '[{"name": "get_weather", "description": "current weather", "parameters": {"city": {}}}, {"name": "get_stock", "description": "a share price", "parameters": {"ticker": {}}}]',
    "answers": '[{"name": "get_weather", "arguments": {"city": "Paris"}}, {"name": "get_weather", "arguments": {"city": "Rome"}}]',
}


def test_a_request_becomes_one_pair_per_offered_tool_labelled_by_whether_it_was_called():
    s = xlam(ROW)
    out = list(table_rows([s], PairPlan()))
    assert [r["label"] for r in out] == [FITS, NO_FIT] and len({r["split"] for r in out}) == 1
    assert "Option: get_weather: current weather (city)\nOther options: get_stock" in out[0]["text"]


def test_some_requests_get_unrelated_tools_so_choosing_none_is_learned():
    pool = [Situation(f"q{i}", [{"name": f"t{i}"}], {f"t{i}"}) for i in range(200)]
    mixed = with_unrelated(pool, random.Random(0))
    assert 10 < sum(not s.chosen for s in mixed) < 60


def test_the_decision_is_right_only_when_exactly_the_called_tools_are_chosen(tmp_path):
    table = tmp_path / "t.csv"
    write(table, [xlam(ROW)] * 50, PairPlan())
    perfect = score(
        table, lambda texts: [0.9 if t.partition("Option: ")[2].startswith("get_weather") else 0.1 for t in texts]
    )
    greedy = score(table, lambda texts: [0.9] * len(texts))
    assert perfect["tool_needed"]["accuracy"] == 1.0 and greedy["tool_needed"]["accuracy"] == 0.0
    assert marked("q", {"name": "x"}, []) == "q\n\nOption: x:  ()\nOther options: none"


def test_a_rival_is_scored_on_the_same_requests_as_the_model(tmp_path, monkeypatch):
    import byosm.llm as llm_module
    from byosm.options import rivals

    class Picks:
        def json(self, system, prompt, schema):
            return {"items": [1]}

    monkeypatch.setattr(llm_module, "get_llm", lambda role, cfg=None: Picks())
    monkeypatch.setattr(llm_module, "role_model", lambda role, cfg=None: "fake:llm")
    table = tmp_path / "t.csv"
    write(table, [xlam(ROW)] * 200, PairPlan())
    out = rivals.rival_scores(table, {"concurrency": 2}, lambda texts: [0.1] * len(texts), sample=10)
    assert out["sample"] == 10 and out["rivals"]["fake:llm"]["tool_needed"]["accuracy"] == 1.0
    assert out["model"]["tool_needed"]["accuracy"] == 0.0


def test_the_options_commands_import_what_they_use():
    from typer.testing import CliRunner

    from byosm.cli import app

    for command in ("score", "rival"):
        assert CliRunner().invoke(app, ["options", command, "--help"]).exit_code == 0


def test_scores_and_comparisons_are_kept_side_by_side(ws):
    from byosm.options.select import report, save

    save(ws, "t", own={"exact": 1})
    save(ws, "t", compare={"sample": 3})
    assert {k: v for k, v in report(ws, "t").items() if k != "at"} == {"own": {"exact": 1}, "compare": {"sample": 3}}


def test_whole_requests_go_to_dev_so_no_request_sits_in_both_train_and_dev():
    pool = [xlam({**ROW, "query": f"weather {i}?"}) for i in range(300)]
    out = list(table_rows(pool, PairPlan()))
    by_group = {}
    for r in out:
        by_group.setdefault(r["group"], set()).add(r["split"])
    assert all(len(s) == 1 for s in by_group.values())
    assert {"train", "dev", "test"} == {s for g in by_group.values() for s in g}


def test_a_table_marking_dev_rows_keeps_them_in_dev(ws, tmp_path):
    import csv

    from byosm.data.local import add_local
    from byosm.spec import Label, TaskSpec

    table = tmp_path / "t.csv"
    with table.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["text", "label", "split"])
        w.writerows([[f"t{i}", "a", "train"] for i in range(20)] + [["d", "a", "dev"], ["x", "a", "test"]])
    spec = TaskSpec(name="q", description="q", modality="text", labels=[Label(name="a", definition="")])
    spec.data.local_dir = str(table)
    spec.data.local_source = "hf:lockon/xlam-function-calling-60k"
    splits = {"train": [], "dev": [], "test": []}
    add_local(spec, splits, random.Random(0), log=lambda m: None)
    assert [r["input"] for r in splits["dev"]] == ["d"] and [r["input"] for r in splits["test"]] == ["x"]
    assert len(splits["train"]) == 20
    assert {r["source"] for rows in splits.values() for r in rows} == {spec.data.local_source}
    before = spec.data_key()
    spec.data.local_source = "local"
    assert spec.data_key() == before
