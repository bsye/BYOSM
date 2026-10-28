from conftest import numbered_items

from byosm.data import judge as judge_module
from byosm.data.judge import DROP, JUDGE_BATCH, judge, judge_labels
from byosm.spec import Label, TaskSpec


def _spec():
    return TaskSpec(name="j", description="", modality="text", labels=[Label(name=n, definition="") for n in "ab"])


class Labeller:
    def json(self, system, prompt, schema):
        items = numbered_items(prompt)
        if "boom" in items:
            raise RuntimeError("down")
        return {"labels": [{"n": i, "label": s if s in "ab" else "__drop__"} for i, s in enumerate(items, 1)]}


def test_the_judge_returns_one_label_per_input_in_order_even_when_a_batch_fails(monkeypatch):
    monkeypatch.setattr(judge_module, "get_llm", lambda role, cfg=None: Labeller())
    texts = ["a", "b", "x"] + ["a"] * (JUDGE_BATCH - 3) + ["boom"] * JUDGE_BATCH + ["b", "a"]
    labels = judge_labels(_spec(), texts, {"concurrency": 2}, log=lambda m: None)
    assert labels[:3] == ["a", "b", DROP] and set(labels[JUDGE_BATCH : 2 * JUDGE_BATCH]) == {None}
    assert labels[2 * JUDGE_BATCH :] == ["b", "a"]


def test_the_judge_keeps_only_rows_whose_label_it_agrees_with(monkeypatch):
    monkeypatch.setattr(judge_module, "get_llm", lambda role, cfg=None: Labeller())
    rows = [{"input": "a", "label": "a"}, {"input": "b", "label": "a"}, {"input": "x", "label": "a"}]
    assert judge(_spec(), rows, {"concurrency": 1}, log=lambda m: None) == rows[:1]
