import pytest
from binary_specs import binary

from byosm.evals.grouped import summary
from byosm.evals.scoring import score
from byosm.options.situations import FITS, NO_FIT
from byosm.predictors import ALWAYS_POSITIVE, get_baseline


def pair(group, i, label, split="test", category="c"):
    meta = {"group": group, "category": category}
    return {"id": f"{group}:{i}", "input": f"{group} {i}", "label": label, "meta": meta, "split": split}


def test_pair_suites_are_also_scored_per_request_and_per_category(tmp_path):
    spec = binary("t", NO_FIT, FITS)
    test = [pair("a", 0, FITS), pair("a", 1, NO_FIT), pair("b", 0, NO_FIT, category="none"), pair("c", 0, FITS)]
    calib = [pair("d", 0, FITS, "calib"), pair("d", 1, NO_FIT, "calib")]
    m = score(get_baseline(ALWAYS_POSITIVE, spec), spec, calib, test, tmp_path)["metrics"]
    found = m["requests"]["all"]
    assert found["no_tool"]["accuracy"] == 0.0 and found["tool_needed"]["accuracy"] == 0.5
    assert m["requests"]["by_category"]["none"]["no_tool"]["n"] == 1 and m["f1"] == pytest.approx(2 * 0.5 / 1.5)
    assert summary(m) == "no tool 0.000, tool needed 0.500"


def test_rows_without_a_request_are_scored_as_plain_rows(tmp_path):
    spec = binary("t", "statement", "question")
    rows = [{"id": "a", "input": "is it?", "label": "question", "meta": {"group": "g"}}]
    assert "requests" not in score(get_baseline(ALWAYS_POSITIVE, spec), spec, [], rows, tmp_path)["metrics"]


def test_the_board_shows_how_many_rows_each_model_was_scored_on(ws, monkeypatch):
    from typer.testing import CliRunner

    from byosm.cli import app
    from byosm.cli.base import console
    from byosm.evals import board_rows
    from byosm.project import Project
    from byosm.spec import EvalSuite

    p = Project("t").ensure()
    suites = {"evals": [EvalSuite(name="bench", primary=True)], "baselines": ["llm"]}
    p.save_spec(binary("t", NO_FIT, FITS).model_copy(update=suites))
    more = {"requests": {"all": {"no_tool": {"accuracy": 0.25}, "tool_needed": {"accuracy": 0.5}}}}
    p.save_result("llm", "bench", {"kind": "baseline", "objective": 0.5, "n_test": 7, "metrics": more})
    assert board_rows("t")[1][0]["bench:n"] == 7
    monkeypatch.setattr(console, "width", 300)
    shown = CliRunner().invoke(app, ["board", "t", "--detail"])
    assert shown.exit_code == 0 and "no tool 0.250, tool needed 0.500" in shown.output and " 7 " in shown.output
