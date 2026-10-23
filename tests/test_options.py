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
