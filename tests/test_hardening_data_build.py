import random

import byosm.data as data
from byosm.data import _split_by_recording, build
from byosm.data.store import Dataset, record
from byosm.spec import Label, TaskSpec


def _clips(files, per_file):
    return [record(f"audio/real/{f}-{i}.wav", [0], "hf:x", file=f) for f in files for i in range(per_file)]


def test_real_clips_from_one_recording_stay_in_one_split():
    parts = _split_by_recording(_clips([f"rec{i}.wav" for i in range(30)], 20), random.Random("vad"))
    train_files = {r["meta"]["file"] for r in parts["train"]}
    dev_files = {r["meta"]["file"] for r in parts["dev"]}
    assert dev_files and train_files
    assert not train_files & dev_files


def test_a_single_recording_is_kept_for_training():
    parts = _split_by_recording(_clips(["only.wav"], 5), random.Random("vad"))
    assert len(parts["train"]) == 5 and not parts["dev"]


def test_panel_rows_that_nearly_copy_a_test_input_do_not_reach_train(tmp_path, monkeypatch):
    spec = TaskSpec(name="t", description="topic", modality="text", labels=[Label(name=n, definition=n) for n in "ab"])
    spec.data.teachers = ["test-keyword"]
    test_text = "the quick brown fox jumps over the lazy dog near the river bank today"
    splits = {
        "train": [record(f"train row number {i}", "a", "local") for i in range(5)],
        "dev": [],
        "calib": [],
        "test": [record(test_text, "a", "local")],
    }
    panel_rows = [record(test_text.upper() + "  ", "b", "panel"), record("something else entirely", "b", "panel")]
    monkeypatch.setattr(data, "_collect", lambda *a: (splits, [], None))
    monkeypatch.setattr(data.panel, "label_pool", lambda *a: panel_rows)
    build(spec, tmp_path, log=lambda _: None)
    train = [r["input"] for r in Dataset(tmp_path).load("train")]
    assert "something else entirely" in train
    assert test_text.upper() + "  " not in train
