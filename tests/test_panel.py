import json

import numpy as np

from byosm.data.panel import PANEL_FILE, combine, label_pool
from byosm.data.review import REVIEWED_SOURCE, reviewed_labels, save_review
from byosm.data.store import read, record
from byosm.events import EVENTS_ENV, parse
from byosm.predictors import Predictor, register_baseline
from byosm.spec import Label, TaskSpec
from byosm.train.teacher import row_soft_targets

LABELS = ["food", "travel"]


@register_baseline("test-keyword", "votes by keyword")
class KeywordTeacher(Predictor):
    def probs_batch(self, raws):
        return np.array([[1.0, 0.0] if "pasta" in r else [0.0, 1.0] for r in raws], np.float32)


@register_baseline("test-coin", "always says food")
class CoinTeacher(Predictor):
    def probs_batch(self, raws):
        return np.tile(np.array([[1.0, 0.0]], np.float32), (len(raws), 1))


def _spec(teachers):
    spec = TaskSpec(
        name="t", description="topic", modality="text", labels=[Label(name=n, definition=n) for n in LABELS]
    )
    spec.data.teachers, spec.data.teacher_rows = teachers, 10
    return spec


def test_the_panel_trusts_the_teacher_that_was_right_on_labelled_rows():
    good, bad = np.array([[0.0, 1.0]]), np.array([[1.0, 0.0]])
    post = combine([good, bad], [0.95, 0.5], np.array([0.5, 0.5]))
    assert post[0, 1] > 0.8
    assert np.allclose(combine([good], [0.95], np.array([0.5, 0.5])).sum(1), 1)


def test_unlabeled_inputs_get_soft_labels_and_a_panel_summary(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("BYOSM_CACHE", str(tmp_path / "cache"))
    monkeypatch.setenv(EVENTS_ENV, "1")
    gold = [record(f"pasta {i}", "food", "local") for i in range(4)] + [
        record(f"bus {i}", "travel", "local") for i in range(4)
    ]
    splits = {"train": gold, "dev": [], "test": []}
    pool = [record(x, None, "hf:x") for x in ("more pasta", "a flight", "a train", "pasta again")]
    rows = label_pool(_spec(["test-keyword", "test-coin"]), splits, pool, tmp_path, print)
    assert [r["label"] for r in rows] and all(abs(sum(r["meta"]["soft"]) - 1) < 1e-3 for r in rows)
    by_input = {r["input"]: r["label"] for r in rows}
    assert by_input["a flight"] == "travel" and by_input["more pasta"] == "food"
    events = [e for e in map(parse, capsys.readouterr().out.splitlines()) if e]
    assert {"kind": "teacher", "teacher": "test-coin", "done": 4, "total": 4, "specialist": "t"} in events
    summary = json.loads((tmp_path / PANEL_FILE).read_text())
    assert summary["teachers"] == {"test-keyword": 1.0, "test-coin": 0.5} and summary["n_pool"] == 4
    assert len(rows) == len(label_pool(_spec(["test-keyword", "test-coin"]), splits, pool, tmp_path, print))


def test_gold_rows_train_on_one_hot_targets_next_to_teacher_rows():
    spec = _spec([])
    rows = [record("x", "travel", "local"), record("y", "food", "teachers", soft=[0.7, 0.3])]
    soft = row_soft_targets(spec, rows)
    assert soft[0].tolist() == [0.0, 1.0] and np.allclose(soft[1], [0.7, 0.3])
    assert row_soft_targets(spec, rows[:1]) is None


def test_a_human_review_overrides_the_panel_with_a_one_hot_label(tmp_path, monkeypatch):
    monkeypatch.setenv("BYOSM_CACHE", str(tmp_path / "cache"))
    gold = [record(f"pasta {i}", "food", "local") for i in range(2)] + [record("bus", "travel", "local")]
    splits = {"train": gold, "dev": [], "test": []}
    pool = [record(x, None, "hf:x") for x in ("more pasta", "a flight")]
    reviewed = {"more pasta": "travel", "a flight": "not a label"}
    rows = label_pool(_spec(["test-keyword"]), splits, pool, tmp_path, print, reviewed)
    by_input = {r["input"]: r for r in rows}
    assert by_input["more pasta"]["label"] == "travel" and by_input["more pasta"]["source"] == REVIEWED_SOURCE
    assert "soft" not in by_input["more pasta"]["meta"] and by_input["a flight"]["source"] != REVIEWED_SOURCE
    targets = row_soft_targets(_spec([]), rows)
    assert targets[[r["input"] for r in rows].index("more pasta")].tolist() == [0.0, 1.0]
    assert json.loads((tmp_path / PANEL_FILE).read_text())["n_reviewed"] == 1


def test_a_second_review_of_the_same_input_replaces_the_first(tmp_path):
    path = tmp_path / "reviewed.jsonl"
    save_review(path, "a", "food")
    save_review(path, "b", "food")
    save_review(path, "a", "travel")
    assert reviewed_labels(path) == {"b": "food", "a": "travel"} and len(read(path)) == 2
