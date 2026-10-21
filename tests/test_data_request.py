import csv

from byosm.config import Project
from byosm.data import _local
from byosm.data.check import check
from byosm.data.request import missing_labels, score_rows_needed, write_request
from byosm.data.table import read_table
from byosm.spec import HFSource, compile_spec

QUESTIONS = [f"what about item {i}?" for i in range(12)]
STATEMENTS = [f"i like item {i}." for i in range(12)]


def spec_without_data(ws):
    spec = compile_spec("q", "detect questions")
    spec.data.synthetic = False
    p = Project("q").ensure()
    spec.save(p.spec_path)
    return p, spec


def write_csv(path, rows):
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["text", "label", "split"])
        w.writerows(rows)
    return path


def test_missing_labels_follow_the_sources(ws):
    _, spec = spec_without_data(ws)
    assert missing_labels(spec) == ["statement", "question"]
    spec.data.hf_sources = [HFSource(dataset="x/y", label_map=[["0", "question"]])]
    assert missing_labels(spec) == ["statement"]
    spec.data.synthetic = True
    assert missing_labels(spec) == []


def test_test_rows_follow_the_confidence_interval():
    total, per_label = score_rows_needed(0.9, 7)
    assert total == 385 and per_label == 55


def test_the_request_writes_a_template_and_the_rows_needed(ws):
    p, spec = spec_without_data(ws)
    out = write_request(p.root, spec, "no data")
    assert read_table(out / "template.xlsx") == read_table(out / "template.csv")
    assert read_table(out / "template.csv")[0] == {"text": "is it on?", "label": "question", "split": ""}
    assert "| question | 100 | 400 |" in (out / "request.md").read_text()


def test_the_check_flags_broken_rows_and_leaks(ws):
    p, spec = spec_without_data(ws)
    rows = [["what time is it?", "question", "test"], ["what time is it?", "question", ""], ["", "question", ""]]
    rows.append(["hello", "greeting", ""])
    report = check(spec, p.root, write_csv(ws / "bad.csv", rows), judge=False)
    kinds = {k for _, k, _ in report["issues"]}
    assert {"empty input", "unknown label", "test row also in training"} <= kinds and report["verdict"] == "fix"


def test_a_clean_table_trains_and_keeps_its_test_rows(ws):
    p, spec = spec_without_data(ws)
    rows = [[q, "question", "test" if i < 3 else ""] for i, q in enumerate(QUESTIONS)]
    rows += [[s, "statement", "test" if i < 3 else ""] for i, s in enumerate(STATEMENTS)]
    path = write_csv(ws / "good.csv", rows)
    report = check(spec, p.root, path, judge=False)
    assert report["verdict"] == "needs more" and report["noise"]["agreement"] > 0.9
    spec.data.local_dir = str(path)
    held = [x for x, _, s in _local(spec) if s == "test"]
    assert len(held) == 6 and (p.root / "data_request" / "report.md").exists()
