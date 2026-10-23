from datasets import Dataset

from byosm.data.hf_source import rows
from byosm.data.hf_source.rows import _row_label
from byosm.spec import HFSource, Label, TaskSpec

SPEC = TaskSpec(
    name="t",
    description="t",
    modality="text",
    labels=[Label(name="negative", definition=""), Label(name="positive", definition="")],
)
OURS = set(SPEC.label_names)


def test_labels_that_already_match_ours_are_kept_without_a_map():
    assert _row_label("positive", None, {}, OURS) == "positive"
    assert _row_label(1, ["negative", "positive"], {}, OURS) == "positive"
    assert _row_label("neutral", None, {}, OURS) is None


def test_a_map_still_decides_when_one_is_given():
    assert _row_label("pos", None, {"pos": "positive"}, OURS) == "positive"
    assert _row_label("positive", None, {"pos": "positive"}, OURS) is None


def test_streamed_rows_keep_their_own_labels_when_no_map_is_given(monkeypatch):
    data = [{"text": f"t{i}", "label": "positive" if i % 2 else "negative"} for i in range(6)]
    data.append({"text": "other", "label": "neutral"})
    monkeypatch.setattr(rows, "open_split", lambda *a: Dataset.from_list(data).to_iterable_dataset())
    src = HFSource(dataset="a/b", input_column="text", label_column="label")
    got = dict(rows._stream_rows(src, SPEC))
    assert sum(1 for y in got.values() if y) == 6
    assert got["other"] is None
