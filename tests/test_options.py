import random

from byosm.options.pairs import FITS, NO_FIT, Situation, pair_text, rows, with_unrelated, write, xlam
from byosm.options.select import score

ROW = {
    "query": "weather in Paris and Rome?",
    "tools": '[{"name": "get_weather", "description": "current weather", "parameters": {"city": {}}}, {"name": "get_stock", "description": "a share price", "parameters": {"ticker": {}}}]',
    "answers": '[{"name": "get_weather", "arguments": {"city": "Paris"}}, {"name": "get_weather", "arguments": {"city": "Rome"}}]',
}


def test_a_request_becomes_one_pair_per_offered_tool_labelled_by_whether_it_was_called():
    s = xlam(ROW)
    out = list(rows([s], random.Random(0)))
    assert [r["label"] for r in out] == [FITS, NO_FIT] and len({r["split"] for r in out}) == 1
    assert "Option: get_weather: current weather (city)" in out[0]["text"]


def test_some_requests_get_unrelated_tools_so_choosing_none_is_learned():
    pool = [Situation(f"q{i}", [{"name": f"t{i}"}], {f"t{i}"}) for i in range(200)]
    mixed = with_unrelated(pool, random.Random(0))
    assert 10 < sum(not s.chosen for s in mixed) < 60


def test_the_decision_is_right_only_when_exactly_the_called_tools_are_chosen(tmp_path):
    table = tmp_path / "t.csv"
    write(table, [xlam(ROW)] * 50)
    perfect = score(table, lambda texts: [0.9 if "get_weather" in t and "Paris" in t else 0.1 for t in texts])
    greedy = score(table, lambda texts: [0.9] * len(texts))
    assert perfect["tool_needed"]["accuracy"] == 1.0 and greedy["tool_needed"]["accuracy"] == 0.0
    assert pair_text("q", {"name": "x"}) == "q\n\nOption: x:  ()"


def test_a_rival_is_scored_on_the_same_requests_as_the_model(tmp_path, monkeypatch):
    import byosm.llm as llm_module
    from byosm.options import rivals

    class Picks:
        def json(self, system, prompt, schema):
            return {"tools": ["get_weather"]}

    monkeypatch.setattr(llm_module, "get_llm", lambda role, cfg=None: Picks())
    monkeypatch.setattr(llm_module, "role_model", lambda role, cfg=None: "fake:llm")
    table = tmp_path / "t.csv"
    write(table, [xlam(ROW)] * 200)
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
