from dataclasses import replace

import pytest
from benchmark_files import BFCL, WHEN2CALL, written
from binary_specs import binary
from typer.testing import CliRunner

from byosm.benchmarks import add as adding
from byosm.benchmarks import bfcl, when2call
from byosm.benchmarks.catalog import BENCHMARKS, fitting
from byosm.benchmarks.pairs import pair_rows, split_of
from byosm.cli import app
from byosm.options.create import spec_for
from byosm.options.situations import FITS, NO_FIT
from byosm.project import Project
from byosm.spec import EvalSuite, PairPlan


def option_name(row):
    return row["input"].partition("Option: ")[2].partition(":")[0]


def test_bfcl_offers_one_pair_per_function_fitting_when_the_answer_calls_it(tmp_path, monkeypatch):
    monkeypatch.setattr(bfcl, "fetch", written(tmp_path, BFCL))
    found = list(bfcl.cases(bfcl.NON_LIVE))
    assert [(c.group, c.situation.chosen) for c in found] == [
        ("simple_0", {"get_weather"}),
        ("multiple_0", {"get_stock"}),
        ("multiple_11", {"get_weather"}),
        ("irrelevance_0", set()),
        ("irrelevance_9", set()),
    ]
    assert found[1].situation.text == "Price of ACME?"
    weather = found[0].situation.options[0]["parameters"]
    assert weather["city"]["required"] and not weather["unit"]["required"]
    rows = pair_rows(binary("t", NO_FIT, FITS), found, bfcl.REPO)
    assert [(r["id"], option_name(r), r["label"]) for r in rows[:3]] == [
        ("simple_0:0", "get_weather", FITS),
        ("multiple_0:0", "get_weather", NO_FIT),
        ("multiple_0:1", "get_stock", FITS),
    ]
    assert all(r["split"] == split_of(r["meta"]["group"]) for r in rows) and {r["split"] for r in rows} == {
        "calib",
        "test",
    }
    assert rows[0]["source"] == f"hf:{bfcl.REPO}" and rows[-1]["meta"]["category"] == "irrelevance"


def test_pairs_are_written_in_the_layouts_the_specialist_is_scored_in(tmp_path, monkeypatch):
    monkeypatch.setattr(bfcl, "fetch", written(tmp_path, BFCL))
    spec = binary("t", NO_FIT, FITS)
    spec.data.pairs = PairPlan(layouts=["json"], held_out_layouts=["prose"])
    rows = bfcl.rows(bfcl.NON_LIVE, spec, print)
    assert {r["meta"]["layout"] for r in rows} <= {"json", "prose"}


def test_when2call_fits_only_the_tool_the_model_should_call_now(tmp_path, monkeypatch):
    monkeypatch.setattr(when2call, "fetch", written(tmp_path, WHEN2CALL))
    rows = when2call.rows(binary("t", NO_FIT, FITS), print)
    assert {(r["meta"]["category"], option_name(r)): r["label"] for r in rows} == {
        ("tool_call", "get_weather"): FITS,
        ("tool_call", "get_stock"): NO_FIT,
        ("cannot_answer", "get_stock"): NO_FIT,
        ("request_for_info", "get_weather"): NO_FIT,
        ("request_for_info", "get_stock"): NO_FIT,
    }
    assert len({r["split"] for r in rows}) == 1 and len({r["meta"]["group"] for r in rows}) == 3


def test_only_benchmarks_that_fit_the_labels_are_offered():
    assert fitting(binary("q", "statement", "question")) == []
    assert {b.name for b in fitting(binary("t", NO_FIT, FITS))} == {"bfcl", "bfcl-live", "when2call"}


def tool_choice(ws, monkeypatch):
    monkeypatch.setattr(bfcl, "fetch", written(ws / "hub", BFCL))
    monkeypatch.setattr(adding, "licence", lambda dataset: "apache-2.0")
    p = Project("tool-choice").ensure()
    spec = spec_for("tool-choice", ws / "t.csv", "lockon/xlam-function-calling-60k", PairPlan())
    p.save_spec(spec.model_copy(update={"baselines": ["llm"]}))
    return p


def test_a_public_suite_is_added_once_its_protocol_check_passes_and_every_rival_is_scored_on_it(ws, monkeypatch):
    from byosm import evals

    p = tool_choice(ws, monkeypatch)
    out = CliRunner().invoke(app, ["suite", "public", "tool-choice", "bfcl"])
    assert out.exit_code == 0, out.output
    suite = next(e for e in p.spec().evals if e.name == "bfcl")
    assert (suite.kind, suite.dataset, suite.licence) == ("local", bfcl.REPO, "apache-2.0")
    report = p.report("protocol:bfcl")
    assert report["passed"] and (report["n_test"], report["n_calib"]) == (4, 3)
    assert "protocol check, no_tool: expected 0.0000, scored 0.0000, ok" in out.output
    assert "byosm eval tool-choice --suite bfcl" in out.output
    scored = evals.run("tool-choice", [], None, ["bfcl"], log=lambda m: None)["llm"]["bfcl"]
    assert scored["n_test"] == 4 and "no_tool" in scored["metrics"]["requests"]["all"]
    again = CliRunner().invoke(app, ["suite", "public", "tool-choice", "bfcl"])
    assert again.exit_code != 0 and "exists" in again.output


def test_a_suite_whose_protocol_check_fails_is_not_added(ws, monkeypatch):
    from byosm.evals.suites import stored

    p = tool_choice(ws, monkeypatch)
    b = BENCHMARKS["bfcl"]
    wrong = replace(b, kind=replace(b.kind, expected=lambda spec, rows: {"f1": 0.123}))
    with pytest.raises(ValueError, match="protocol check"):
        adding.add(p, wrong, lambda m: None)
    assert [e.name for e in p.spec().evals] == ["builtin"] and not p.report("protocol:bfcl")["passed"]
    assert stored(p, p.spec(), EvalSuite(name="bfcl", kind="local")) is None


def test_unknown_or_unfitting_benchmarks_are_refused(ws, monkeypatch):
    p = tool_choice(ws, monkeypatch)
    with pytest.raises(ValueError, match="no public benchmark"):
        adding.chosen(p, ["toolret"])
    p.save_spec(binary("tool-choice", "statement", "question"))
    with pytest.raises(ValueError, match="does not fit"):
        adding.chosen(p, ["bfcl"])
