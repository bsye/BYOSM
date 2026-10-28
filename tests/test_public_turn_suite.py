import pytest
from benchmark_files import EOT_HOLDS, EOT_TURNS, eot_summary
from binary_specs import binary
from typer.testing import CliRunner

from byosm.benchmarks import add as adding
from byosm.benchmarks import eot
from byosm.benchmarks.catalog import BENCHMARKS, fitting
from byosm.benchmarks.source import split_of
from byosm.cli import app
from byosm.project import Project


def test_each_pause_becomes_the_texts_heard_while_it_lasts_with_eot_bench_transcript_lag():
    words = EOT_TURNS[0]["words"]
    assert list(eot.states(words, 0.8, 1.3)) == [(0.0, "hello"), (0.5, "hello there")]
    assert list(eot.states(words, 2.2, 3.4)) == [
        (0.0, "hello there how"),
        (0.1, "hello there how are"),
        (0.5, "hello there how are you"),
    ]


def test_only_the_spans_eot_bench_scores_become_rows():
    rows = eot.turn_rows(EOT_TURNS[0], "finished", "unfinished", "hf:x")
    spans = {(r["meta"]["span"], r["meta"]["span_s"], r["label"]) for r in rows}
    assert spans == {("en_a:0", 0.5, "unfinished"), ("en_a:2", 1.2, "finished")}
    assert {r["split"] for r in rows} == {split_of("en_a")} and rows[0]["meta"]["at_s"] == 0.0
    short = eot.turn_rows(EOT_TURNS[1], "finished", "unfinished", "hf:x")
    assert {r["meta"]["span_s"] for r in short if r["label"] == "unfinished"} == {0.25}


def timeout_cutoff(budget_s):
    return sum(h > budget_s for h in EOT_HOLDS) / len(EOT_HOLDS)


def turn_completion(ws, monkeypatch, at_300=None):
    published = eot_summary("en", at_300 or timeout_cutoff(0.3), timeout_cutoff(0.6))
    monkeypatch.setattr(eot, "turns", lambda language: EOT_TURNS)
    monkeypatch.setattr(eot, "summary", lambda: published)
    monkeypatch.setattr(adding, "licence", lambda dataset: "cc-by-4.0")
    p = Project("turn-completion").ensure()
    p.save_spec(binary("turn-completion", "unfinished", "finished", "recall_at_precision"))
    return p


def test_eot_bench_is_added_once_always_finished_reproduces_its_published_silence_timeout(ws, monkeypatch):
    p = turn_completion(ws, monkeypatch)
    out = CliRunner().invoke(app, ["suite", "public", "turn-completion"])
    assert out.exit_code == 0, out.output
    report = p.report("protocol:eot-bench")
    assert report["passed"] and report["published"]["LiveKit Turn Detector v1"]["cutoff at 300 ms"] == 0.05
    assert "published on this split, LiveKit Turn Detector v1" in out.output
    suite = p.spec().evals[-1]
    assert (suite.name, suite.kind, suite.dataset, suite.licence) == ("eot-bench", "local", eot.REPO, "cc-by-4.0")


def test_eot_bench_is_refused_when_its_silence_timeout_is_not_reproduced(ws, monkeypatch):
    p = turn_completion(ws, monkeypatch, at_300=0.5)
    out = CliRunner().invoke(app, ["suite", "public", "turn-completion", "eot-bench"])
    assert out.exit_code != 0 and "protocol check" in out.output
    assert [e.name for e in p.spec().evals] == ["builtin"] and "published" not in p.report("protocol:eot-bench")


def test_a_language_eot_bench_publishes_no_baseline_for_is_refused(ws, monkeypatch):
    p = turn_completion(ws, monkeypatch)
    monkeypatch.setattr(eot, "summary", lambda: eot_summary("de", 0.5, 0.2))
    with pytest.raises(ValueError, match="publishes no silence-timeout baseline"):
        adding.add(p, BENCHMARKS["eot-bench"], lambda m: None)


def test_eot_bench_fits_turn_specialists_in_its_languages_only():
    turn = binary("t", "unfinished", "finished")
    assert [b.name for b in fitting(turn)] == ["eot-bench"]
    assert fitting(binary("q", "statement", "question")) == []
    assert fitting(turn.model_copy(update={"languages": ["sv"]})) == []
